import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "best.pt"
model = YOLO(model_path)


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    annotated_frame = results[0].plot()
  
    cv2.imshow('YOLO Live Detection', annotated_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
