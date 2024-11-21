import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay

# Define screen dimensions
screen_width = 1920
screen_height = 1080

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Calibration points (to be updated during calibration)
calibration_points = {
    'top_left': [0, 0],
    'top_center': [screen_width // 2, 0],
    'top_right': [screen_width, 0],
    'center_left': [0, screen_height // 2],
    'center': [screen_width // 2, screen_height // 2],
    'center_right': [screen_width, screen_height // 2],
    'bottom_left': [0, screen_height],
    'bottom_center': [screen_width // 2, screen_height],
    'bottom_right': [screen_width, screen_height]
}

# Iris calibration points (will be updated during calibration)
iris_calibration_points = {
    'top_left': [0, 0],
    'top_center': [0, 0],
    'top_right': [0, 0],
    'center_left': [0, 0],
    'center': [0, 0],
    'center_right': [0, 0],
    'bottom_left': [0, 0],
    'bottom_center': [0, 0],
    'bottom_right': [0, 0]
}

# Prepare Delaunay triangulation
calibration_list = list(calibration_points.values())
tri = Delaunay(calibration_list)

def calculate_barycentric(p, triangle):
    """
    Calculate barycentric coordinates for a point within a given triangle.
    """
    A = np.array(triangle)
    B = np.array([p[0], p[1], 1])
    T = np.array([
        [A[0][0], A[1][0], A[2][0]],
        [A[0][1], A[1][1], A[2][1]],
        [1, 1, 1]
    ])
    try:
        inv_T = np.linalg.inv(T)
        barycentric_coords = inv_T.dot(B)
        return barycentric_coords
    except np.linalg.LinAlgError:
        return None

def map_to_screen(iris_center):
    """
    Map the iris center to screen coordinates using Delaunay triangulation and barycentric interpolation.
    """
    # Find the triangle containing the iris center
    simplex = tri.find_simplex(iris_center)
    if simplex == -1:
        return None  # Point is outside the triangulation

    # Get vertices of the triangle
    vertices = tri.simplices[simplex]
    triangle = [calibration_list[i] for i in vertices]

    # Get the barycentric coordinates of the iris center with respect to the triangle
    barycentric_coords = calculate_barycentric(iris_center, triangle)
    if barycentric_coords is None:
        return None

    # Interpolate screen coordinates using barycentric coordinates
    screen_coords = np.dot(barycentric_coords, np.array([calibration_list[i] for i in vertices]))
    return int(screen_coords[0]), int(screen_coords[1])

def calibrate_point(point_name, iris_center):
    """
    Update the calibration point based on the current iris center.
    """
    iris_calibration_points[point_name] = iris_center
    print(f"Calibrated {point_name}: {iris_center}")

# Function to display pointer on the screen
def display_pointer(screen_x, screen_y):
    # Create a blank screen (black background)
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Define pointer properties
    pointer_radius = 10  # Radius of the circle
    pointer_color = (0, 255, 0)  # Green color in BGR
    pointer_thickness = -1  # Filled circle

    # Draw the pointer at the calculated screen position
    if 0 <= screen_x < screen_width and 0 <= screen_y < screen_height:
        cv2.circle(screen, (screen_x, screen_y), pointer_radius, pointer_color, pointer_thickness)
    else:
        print("Pointer coordinates out of screen bounds!")

    # Show the screen with the pointer
    cv2.imshow("Gaze Pointer", screen)
    cv2.waitKey(1)

#Main Eye-Tracking Loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the iris center (Mediapipe iris landmarks: 474 and 475 for right eye)
            iris_x = int(face_landmarks.landmark[474].x * w)
            iris_y = int(face_landmarks.landmark[474].y * h)
            iris_center = [iris_x, iris_y]

            # Map the iris center to screen coordinates
            screen_coords = map_to_screen(iris_center)
            if screen_coords:
                x_screen, y_screen = screen_coords
                cv2.circle(frame, (x_screen, y_screen), 10, (0, 255, 0), -1)

            # Display the detected iris center
            cv2.circle(frame, (iris_x, iris_y), 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Eye Tracking", frame)
    key = cv2.waitKey(1)

    
    # Press 'c' to calibrate points (example)
    if key == ord('c'):
        calibrate_point('center', iris_center)

    # Press 'Esc' to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
