import cv2
import socket
import base64

# Capture video from camera
cap = cv2.VideoCapture(0)

# Set up socket
host = '127.0.0.1'  # Localhost
port = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
print("Socket is listening...")

conn, addr = s.accept()
print("Connected by", addr)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Encode frame and send to Unity
    encoded, buffer = cv2.imencode('.jpg', frame)
    message = base64.b64encode(buffer).decode('utf-8')
    message = message + '\n'  # Add a delimiter
    conn.sendall(message.encode('utf-8'))
