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
                
                # Calibration logic
                if calibration_points['left'] is None or calibration_points['right'] is None or calibration_points['top'] is None or calibration_points['bottom'] is None:
                    # Assuming some condition or user input to set the calibration points
                    print("Calibration mode: Adjust your position to the corners of the screen.")
                    
                    # For demonstration, we'll assume 'calibration' key press will set points
                    cv2.putText(frame_bgr, "Press 'c' to calibrate the extreme points", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        if calibration_points['left'] is None:
                            calibration_points['left'] = iris_center
                            print("Left boundary calibrated.")
                            
                        elif calibration_points['right'] is None:
                            calibration_points['right'] = iris_center
                            print("Right boundary calibrated.")
                        elif calibration_points['top'] is None:
                            calibration_points['top'] = iris_center
                            print("Top boundary calibrated.")
                        elif calibration_points['bottom'] is None:
                            calibration_points['bottom'] = iris_center
                            print("Bottom boundary calibrated.")
                        
                if all(v is not None for v in calibration_points.values()):
                    # Map the central point to screen coordinates using calibration
                    x_range = calibration_points['right'][0] - calibration_points['left'][0]
                    y_range = calibration_points['bottom'][1] - calibration_points['top'][1]
                    
                    x_screen = int(((iris_center[0] - calibration_points['left'][0]) / x_range) * screen_width)
                    y_screen = int(((iris_center[1] - calibration_points['top'][1]) / y_range) * screen_height)
                    
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
