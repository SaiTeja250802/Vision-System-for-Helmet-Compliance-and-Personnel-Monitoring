import cv2 as cv
import numpy as np
import dlib
from ultralytics import YOLO

# Paths
haar_cascade_path = r'D:\learn\ML\OpenCV\face detection vscode\haar_face.xml'
face_recognizer_path = r'D:\learn\ML\OpenCV\face detection vscode\face_trained.yml'
shape_predictor_path = r'D:\learn\ML\OpenCV\shape_predictor_68_face_landmarks.dat'
people_path = 'people.npy'
model_path = "best.pt"

# Load the list of people
people = np.load(people_path).tolist()

# Initialize the Haar cascade and LBPH face recognizer
haar_cascade = cv.CascadeClassifier(haar_cascade_path)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(face_recognizer_path)

# Initialize Dlib's face detector (HOG + SVM) and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Load the trained YOLO model
model = YOLO(model_path)


# Function to recognize faces within detected bounding boxes
def recognize_faces(frame, faces):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    recognized_faces = []

    for (x, y, w, h) in faces:
        # Ensure the bounding box stays within the frame bounds
        x, y = max(0, x), max(0, y)
        w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

        face_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face_roi)
        recognized_faces.append((x, y, w, h, people[label], confidence))

    return recognized_faces


# Function to use webcam for real-time face recognition
def real_time_face_recognition():
    cap = cv.VideoCapture(0)  # Use 0 for built-in webcam, 1 for external

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform detection using YOLO
        results = model(frame)

        # Extract detected faces and other objects
        detected_faces = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                detected_faces.append((x, y, w, h))

        # Recognize faces within the detected bounding boxes
        recognized_faces = recognize_faces(frame, detected_faces)

        # Annotate the frame with YOLO detection results
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                class_id = int(box.cls.tolist()[0])  # Convert tensor to int
                confidence = box.conf.tolist()[0]  # Confidence of the detection
                label = model.names[class_id]  # Convert class index to label

                # Draw the YOLO detection
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0),
                           2)

        # Annotate the frame with face recognition results
        for (x, y, w, h, name, confidence) in recognized_faces:
            cv.putText(frame, f'{name} ({confidence:.2f})', (x, y + h + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),
                       2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('YOLO + Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Run real-time face recognition
real_time_face_recognition()
