import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import pyautogui
import mediapipe as mp
import csv
import pdb
import time
import math
import random

# Initialize Mediapipe FaceMesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

print(f'Screen Width: {screen_width}, Screen Hight: {screen_height}')

# Calibration variables
calibration_points = {'left': None, 'right': None, 'top': None, 'bottom': None}
iris_calibration = {'left': None, 'right': None, 'top': None, 'bottom': None}
center_calibration = {'x': None, 'y': None}  # New calibration for iris center
extra_calibration = {'left': None, 'right': None, 'top': None, 'bottom': None}

# Flags for calibration mode
screen_calibration_mode = True
iris_calibration_mode = False
center_calibration_mode = False  # New flag for center calibration
extra_calibration_mode = False

# Threshold for Gaze Smoothness 
PROXIMITY_THRESHOLD = 300.0
# Threshold for blink detection
EAR_THRESHOLD = 0.2
old_coordinates = (800, 800)
proximity_coords = []

def distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    # Calculate the distances between the vertical eye landmarks
    vertical_1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    vertical_2 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    # Calculate the distance between the horizontal eye landmarks
    horizontal = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Open CSV file for writing coordinates
with open('coordinates.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['x_screen', 'y_screen', 'blink_detected'])  # Write header row

    # Configure FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        # Initialize Tkinter
        root = tk.Tk()
        root.overrideredirect(True)  # Remove the window border
        root.attributes("-topmost", True)  # Keep the window on top

        # Load and check the eye image
        try:
            eye_image = Image.open("eye.png")
            eye_image = eye_image.resize((100, 100), Image.LANCZOS)  # Resize the image to fit your needs
            eye_photo = ImageTk.PhotoImage(eye_image)
        except Exception as e:
            print(f"Error loading image: {e}")
            root.destroy()
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Create a label to display the image
        # label = tk.Label(root, image=eye_photo, bg='black')
        # label.pack()

        previous_iris_center = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the image color from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find facial landmarks
            results = face_mesh.process(frame_rgb)

            # Convert back to BGR for rendering with OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh annotations on the image
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    if cv2.waitKey(1) & 0xFF == ord('d'):
                        pdb.set_trace()

                    # Extract relevant landmarks
                    left_iris_landmarks = [face_landmarks.landmark[i] for i in range(474, 478)]
                    right_iris_landmarks = [face_landmarks.landmark[i] for i in range(469, 473)]
                    
                     # Extract eye landmarks (example indices, adjust as needed)
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]

                    # Calculate the central point of the iris landmarks
                    def get_center(landmarks):
                        x_coords = [lm.x for lm in landmarks]
                        y_coords = [lm.y for lm in landmarks]
                        return np.mean(x_coords), np.mean(y_coords)

                    iris_center = get_center(left_iris_landmarks + right_iris_landmarks)

                    if previous_iris_center is not None:
                        # Check for significant deviation
                        dx = abs(iris_center[0] - previous_iris_center[0])
                        dy = abs(iris_center[1] - previous_iris_center[1])
                        if dx > 0.05 or dy > 0.05:
                            print("Significant deviation detected, recalibrating.")
                            screen_calibration_mode = True
                            iris_calibration_mode = False
                            center_calibration_mode = False  # Reset center calibration
                            extra_calibration_mode = False

                    previous_iris_center = iris_center

                    # Calibration mode indicators
                    if screen_calibration_mode:
                        cv2.putText(frame_bgr, "Press 'c' to calibrate screen boundaries", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            if calibration_points['left'] is None:
                                calibration_points['left'] = iris_center
                                print("Screen left boundary calibrated.")
                            elif calibration_points['right'] is None:
                                calibration_points['right'] = iris_center
                                print("Screen right boundary calibrated.")
                            elif calibration_points['top'] is None:
                                calibration_points['top'] = iris_center
                                print("Screen top boundary calibrated.")
                            elif calibration_points['bottom'] is None:
                                calibration_points['bottom'] = iris_center
                                print("Screen bottom boundary calibrated.")
                                screen_calibration_mode = False
                                iris_calibration_mode = True
                                print("Calibration Points:", calibration_points)
                                print("Switching to iris calibration mode.")

                    elif iris_calibration_mode:
                        cv2.putText(frame_bgr, "Press 'i' to calibrate iris ball tracking", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if cv2.waitKey(1) & 0xFF == ord('i'):
                            if iris_calibration['left'] is None:
                                iris_calibration['left'] = iris_center
                                print("Iris left boundary calibrated.")
                            elif iris_calibration['right'] is None:
                                iris_calibration['right'] = iris_center
                                print("Iris right boundary calibrated.")
                            elif iris_calibration['top'] is None:
                                iris_calibration['top'] = iris_center
                                print("Iris top boundary calibrated.")
                            elif iris_calibration['bottom'] is None:
                                iris_calibration['bottom'] = iris_center
                                print("Iris bottom boundary calibrated.")
                                iris_calibration_mode = False
                                extra_calibration_mode = True
                                print("Iris Calibration Points:", iris_calibration)
                                print("Switching to extra calibration mode.")

                    elif extra_calibration_mode:
                        cv2.putText(frame_bgr, "Press 'e' to calibrate extra boundaries", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if cv2.waitKey(1) & 0xFF == ord('e'):
                            if extra_calibration['left'] is None:
                                extra_calibration['left'] = iris_center
                                print("Extra left boundary calibrated.")
                            elif extra_calibration['right'] is None:
                                extra_calibration['right'] = iris_center
                                print("Extra right boundary calibrated.")
                            elif extra_calibration['top'] is None:
                                extra_calibration['top'] = iris_center
                                print("Extra top boundary calibrated.")
                            elif extra_calibration['bottom'] is None:
                                extra_calibration['bottom'] = iris_center
                                print("Extra bottom boundary calibrated.")
                                extra_calibration_mode = False
                                center_calibration_mode = True
                                print("Extra Calibration Points:", extra_calibration)
                                print("Switching to center calibration mode.")

                    # New Center Calibration Mode
                    elif center_calibration_mode:
                        cv2.putText(frame_bgr, "Press 'x' to calibrate center point", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if cv2.waitKey(1) & 0xFF == ord('x'):
                            calibration_points['center'] = iris_center
                            print("Center point calibrated:", calibration_points['center'])
                            center_calibration_mode = False
                            print("Calibration complete. All points:", calibration_points, iris_calibration, extra_calibration)


                    # Move the cursor based on calibration if all calibrations are complete
                    if (all(v is not None for v in calibration_points.values()) and all(v is not None for v in iris_calibration.values()) and all(v is not None for v in extra_calibration.values()) and calibration_points.get('center') is not None):

                        x_range = calibration_points['right'][0] - calibration_points['left'][0]
                        y_range = calibration_points['bottom'][1] - calibration_points['top'][1]
                        
                        # Map iris center to screen coordinates
                        x_screen = int(((iris_center[0] - calibration_points['left'][0]) / x_range) * screen_width)
                        y_screen = int(((iris_center[1] - calibration_points['top'][1]) / y_range) * screen_height)
                        
                        iris_x_range = iris_calibration['right'][0] - iris_calibration['left'][0]
                        iris_y_range = iris_calibration['bottom'][1] - iris_calibration['top'][1]
                        
                        # Adjust the x and y screen coordinates considering iris movement
                        x_screen_iris = int(((iris_center[0] - iris_calibration['left'][0]) / iris_x_range) * screen_width)
                        y_screen_iris = int(((iris_center[1] - iris_calibration['top'][1]) / iris_y_range) * screen_height)
                        
                        extra_x_range = extra_calibration['right'][0] - extra_calibration['left'][0]
                        extra_y_range = extra_calibration['bottom'][1] - extra_calibration['top'][1]
                        
                        # Adjust the x and y screen coordinates considering iris movement
                        x_screen_extra = int(((iris_center[0] - extra_calibration['left'][0]) / extra_x_range) * screen_width)
                        y_screen_extra = int(((iris_center[1] - extra_calibration['top'][1]) / extra_y_range) * screen_height)
                        
                        # alpha = 0.2  # Smoothing factor (lower values = more smoothing)
                        # x_screen = int(alpha * x_screen + (1 - alpha) * old_coordinates[0])
                        # y_screen = int(alpha * y_screen + (1 - alpha) * old_coordinates[1])
                        
                        # Smooth cursor movement
                        x_screen = int(0.4 * x_screen + 0.5 * x_screen_iris + 0.3 * x_screen_extra)
                        y_screen = int(0.4 * y_screen + 0.5 * y_screen_iris + 0.3 * y_screen_extra)
                        

                        # Update old_coordinates with the smoothed values
                        old_coordinates = (x_screen, y_screen)
                        
                        #time.sleep(0.05)
                        new_coordinates = (x_screen,y_screen)
                        if distance(old_coordinates, new_coordinates) > PROXIMITY_THRESHOLD:
                            old_coordinates = new_coordinates
                            proximity_coords = [new_coordinates]                            
                        else:
                            proximity_coords.append(new_coordinates)
                            if len(proximity_coords)> 9:
                                x_screen = int(sum(coord[0] for coord in proximity_coords[-9:]) / len(proximity_coords[-9:]))
                                y_screen = int(sum(coord[1] for coord in proximity_coords[-9:]) / len(proximity_coords[-9:]))
                            else:
                                x_screen = int(sum(coord[0] for coord in proximity_coords) / len(proximity_coords))
                                y_screen = int(sum(coord[1] for coord in proximity_coords) / len(proximity_coords))
                                
                            if x_screen > 1800:
                                x_screen = 1800
                                
                            if x_screen < 0:
                                x_screen = 0
                            
                            if y_screen > 1100:
                                y_screen = 1100
                                
                            if y_screen <0:
                                y_screen = 0
                        
                        # Calculate EAR for both eyes
                        left_ear = calculate_ear(left_eye_landmarks)
                        right_ear = calculate_ear(right_eye_landmarks)
                        
                        # Check if blink occurred
                        blink_detected = (left_ear < EAR_THRESHOLD) and (right_ear < EAR_THRESHOLD)
                            
                        #print(x_screen,y_screen)
                        # Update the Tkinter window
                        root.geometry(f"{eye_image.width}x{eye_image.height}+{x_screen}+{y_screen}")  # Set the window size and position
                        #root.geometry(f"{x_screen}+{y_screen}")  # Set the window size and position
                        root.update()

                        # Write coordinates to CSV file
                        print(f'X Coordinate: {x_screen}, Y Coordinate: {y_screen}, Blink Detected: {blink_detected}')
                        csv_writer.writerow([x_screen, y_screen,blink_detected])
                        
                    # Draw circles around the landmarks for visualization
                    for landmark in left_iris_landmarks + right_iris_landmarks:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)

                    # Display the image with annotations
                    #resized_image = cv2.resize(frame_bgr, (screen_width-20, screen_height-100))
                    cv2.imshow('Mediapipe Iris and Face Landmarks', frame_bgr)
                    #cv2.imshow('Mediapipe Iris and Face Landmarks', resized_image)

                    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                        root.destroy()  # Destroy the Tkinter window
                        break

        cap.release()
        cv2.destroyAllWindows()