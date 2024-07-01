import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "best.pt"
model = YOLO(model_path)


# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam, 1 for external

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection on the current frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLO Live Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
