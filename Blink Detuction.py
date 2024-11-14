import csv
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

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

# Threshold for blink detection
EAR_THRESHOLD = 0.2

# Process each frame (assuming you have a loop to process video frames)
for frame in video_frames:
    # Get face landmarks
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract iris landmarks
            left_iris_landmarks = [face_landmarks.landmark[i] for i in range(474, 478)]
            right_iris_landmarks = [face_landmarks.landmark[i] for i in range(469, 473)]
            
            # Extract eye landmarks (example indices, adjust as needed)
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            
            # Check if blink occurred
            blink_detected = (left_ear < EAR_THRESHOLD) and (right_ear < EAR_THRESHOLD)
            
            # Your existing gaze point calculation code here...
            
            # Write gaze point and blink status to CSV
            csv_writer.writerow([x_screen, y_screen, blink_detected])
