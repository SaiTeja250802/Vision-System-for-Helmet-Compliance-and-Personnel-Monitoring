import os
import cv2 as cv
import numpy as np
import dlib

# Paths
dataset_path = r'D:\learn\ML\OpenCV\dataset'
haar_cascade_path = r'D:\learn\ML\OpenCV\face detection vscode\haar_face.xml'
face_recognizer_path = r'D:\learn\ML\OpenCV\face detection vscode\face_trained.yml'
shape_predictor_path = r'D:\learn\ML\OpenCV\shape_predictor_68_face_landmarks.dat'

# Initialize the Haar cascade and LBPH face recognizer
haar_cascade = cv.CascadeClassifier(haar_cascade_path)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Initialize Dlib's face detector (HOG + SVM) and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


# Function to create training data
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
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces = detector(gray)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces_regionofinterest = gray[y:y + h, x:x + w]
                features.append(faces_regionofinterest)
                labels.append(label)

    return features, labels, people


# Create training data
features, labels, people = create_training_data()

# Convert lists to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Train the recognizer
face_recognizer.train(features, labels)

# Save the trained model and people list
face_recognizer.save(face_recognizer_path)
np.save('people.npy', people)

print('Training completed and model saved.')
