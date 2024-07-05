
import os
import cv2 as cv
import numpy as np
import dlib
from sklearn.model_selection import train_test_split

dataset_path = r'D:\learn\ML\OpenCV\dataset'
haar_cascade_path = r'D:\learn\ML\OpenCV\face detection vscode\haar_face.xml'
face_recognizer_path = r'D:\learn\ML\OpenCV\face detection vscode\face_trained.yml'
shape_predictor_path = r'D:\learn\ML\OpenCV\shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    return gray

def create_training_data():
    features = []
    labels = []
    people = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        label = len(people)
        people.append(person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img_array = cv.imread(img_path)
            gray = preprocess_image(img_array)

            faces = detector(gray)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_roi = gray[y:y + h, x:x + w]
                features.append(face_roi)
                labels.append(label)

    return features, labels, people

features, labels, people = create_training_data()

features = np.array(features, dtype='object')
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

radius = 1
neighbors = 8
grid_x = 8
grid_y = 8
face_recognizer = cv.face.LBPHFaceRecognizer_create(radius, neighbors, grid_x, grid_y)

face_recognizer.train(X_train, y_train)

val_predictions = []
for img in X_val:
    label, confidence = face_recognizer.predict(img)
    val_predictions.append(label)

val_accuracy = np.mean(val_predictions == y_val)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

face_recognizer.save(face_recognizer_path)
np.save('people1.npy', people)

test_predictions = []
for img in X_test:
    label, confidence = face_recognizer.predict(img)
    test_predictions.append(label)

test_accuracy = np.mean(test_predictions == y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

print('Training completed, model saved, and evaluation done.')

