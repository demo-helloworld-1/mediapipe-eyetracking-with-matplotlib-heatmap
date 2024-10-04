import cv2

# Attempt to open the camera connected via HDMI port
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: HDMI Camera not working")
else:
    print("HDMI Camera is working")
