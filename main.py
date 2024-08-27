import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize Mediapipe FaceMesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Calibration variables
calibration_points = {'left': None, 'right': None, 'top': None, 'bottom': None}
iris_calibration = {'left': None, 'right': None, 'top': None, 'bottom': None}

# Flags for calibration mode
screen_calibration_mode = True
iris_calibration_mode = False

# Configure FaceMesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
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
                
                # Extract relevant landmarks
                left_iris_landmarks = [face_landmarks.landmark[i] for i in range(474, 478)]
                right_iris_landmarks = [face_landmarks.landmark[i] for i in range(469, 473)]
                
                # Calculate the central point of the iris landmarks
                def get_center(landmarks):
                    x_coords = [lm.x for lm in landmarks]
                    y_coords = [lm.y for lm in landmarks]
                    return np.mean(x_coords), np.mean(y_coords)
                
                iris_center = get_center(left_iris_landmarks + right_iris_landmarks)
                
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
                            print("Calibration complete.")
                    
                # Move cursor based on calibration if both calibrations are complete
                if all(v is not None for v in calibration_points.values()) and all(v is not None for v in iris_calibration.values()):
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
                    
                    # Smooth cursor movement
                    x_screen = int(0.7 * x_screen + 0.5 * x_screen_iris)
                    y_screen = int(0.7 * y_screen + 0.5 * y_screen_iris)
                    
                    # Move the cursor
                    pyautogui.moveTo(x_screen, y_screen)
                
                # Draw circles around the landmarks for visualization
                for landmark in left_iris_landmarks + right_iris_landmarks:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)
        
        # Display the image with annotations
        cv2.imshow('Mediapipe Iris and Face Landmarks', frame_bgr)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
