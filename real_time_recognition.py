import cv2 as cv
import numpy as np
import dlib

# Paths
haar_cascade_path = r'D:\learn\ML\OpenCV\face detection vscode\haar_face.xml'
face_recognizer_path = r'D:\learn\ML\OpenCV\face detection vscode\face_trained.yml'
shape_predictor_path = r'D:\learn\ML\OpenCV\shape_predictor.dat'
people_path = 'people.npy'

# Load the list of people
people = np.load(people_path).tolist()

# Initialize the Haar cascade and LBPH face recognizer
haar_cascade = cv.CascadeClassifier(haar_cascade_path)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(face_recognizer_path)

# Initialize Dlib's face detector (HOG + SVM) and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


# Function to use webcam for real-time face recognition
def real_time_face_recognition():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            faces_regionofinterest = gray[y:y + h, x:x + w]
            label, confidence = face_recognizer.predict(faces_regionofinterest)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.putText(frame, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0,
                       (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('Detected face', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Run real-time face recognition
real_time_face_recognition()
