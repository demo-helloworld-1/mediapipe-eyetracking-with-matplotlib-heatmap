import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    
    frame_h, frame_w, _ = frame.shape
    if landmarks_points:
        landmarks = landmarks_points[0].landmark
        
        # Detecting the eyes
        left_eye_x = []
        right_eye_x = []
        for landmark in landmarks[474:478]:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            left_eye_x.append(x)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
        
        for landmark in landmarks[469:473]:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            right_eye_x.append(x)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        
        # Detecting the nose bridge
        nose_bridge_x = []
        for landmark in landmarks[1:2]:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            nose_bridge_x.append(x)
        
        # Determine gaze direction
        if left_eye_x and right_eye_x and nose_bridge_x:
            left_eye_center = sum(left_eye_x) / len(left_eye_x)
            right_eye_center = sum(right_eye_x) / len(right_eye_x)
            nose_bridge_center = sum(nose_bridge_x) / len(nose_bridge_x)
            
            eye_center = (left_eye_center + right_eye_center) / 2
            
            if eye_center < nose_bridge_center - 10:
                direction = "Looking Right"
            elif eye_center > nose_bridge_center + 5:
                direction = "Looking Left"
            else:
                direction = "Looking Center"
            
            # Print the direction on the frame
            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Eye Controlled Mouse", frame)
    
    # Check for 'Q' or 'ESC' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is the ESC key
        break

cam.release()
cv2.destroyAllWindows()
