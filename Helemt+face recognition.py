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

people = np.load(people_path).tolist()

haar_cascade = cv.CascadeClassifier(haar_cascade_path)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(face_recognizer_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

model = YOLO(model_path)


def recognize_faces(frame, faces):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    recognized_faces = []

    for (x, y, w, h) in faces:
        x, y = max(0, x), max(0, y)
        w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

        face_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face_roi)
        recognized_faces.append((x, y, w, h, people[label], confidence))

    return recognized_faces

def real_time_face_recognition():
    cap = cv.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)

        detected_faces = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                detected_faces.append((x, y, w, h))

        recognized_faces = recognize_faces(frame, detected_faces)

        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh.tolist()[0]
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                class_id = int(box.cls.tolist()[0]) 
                confidence = box.conf.tolist()[0] 
                label = model.names[class_id]  

                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0),
                           2)

        for (x, y, w, h, name, confidence) in recognized_faces:
            cv.putText(frame, f'{name} ({confidence:.2f})', (x, y + h + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),
                       2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('YOLO + Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


real_time_face_recognition()
