
import cv2 as cv
import numpy as np
import dlib

# Paths
haar_cascade_path = r'D:\learn\ML\OpenCV\face detection vscode\haar_face.xml'
face_recognizer_path = r'D:\learn\ML\OpenCV\face detection vscode\face_trained.yml'
shape_predictor_path = r'D:\learn\ML\OpenCV\shape_predictor_68_face_landmarks.dat'
people_path = 'people1.npy'

people = np.load(people_path).tolist()

radius = 1
neighbors = 8
grid_x = 8
grid_y = 8
face_recognizer = cv.face.LBPHFaceRecognizer_create(radius, neighbors, grid_x, grid_y)
face_recognizer.read(face_recognizer_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    return gray



def align_face(gray, landmarks):
    eye_left = (landmarks.part(36).x, landmarks.part(36).y)
    eye_right = (landmarks.part(45).x, landmarks.part(45).y)

    dY = eye_right[1] - eye_left[1]
    dX = eye_right[0] - eye_left[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    desired_right_eye_x = 1.0 - 0.35
    desired_dist_between_eyes = desired_right_eye_x - 0.35
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = desired_dist_between_eyes * 256
    scale = desired_dist / dist

    eyes_center = ((eye_left[0] + eye_right[0]) // 2, (eye_left[1] + eye_right[1]) // 2)

    M = cv.getRotationMatrix2D(eyes_center, angle, scale)
    tX = 256 * 0.5
    tY = 256 * 0.35
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    output = cv.warpAffine(gray, M, (256, 256), flags=cv.INTER_CUBIC)
    return output


def real_time_face_recognition():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = preprocess_image(frame)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y + h, x:x + w]

            landmarks = predictor(gray, face)

            aligned_face = align_face(gray, landmarks)

            label, confidence = face_recognizer.predict(aligned_face)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            for n in range(0, 68):
                x_lm = landmarks.part(n).x
                y_lm = landmarks.part(n).y
                cv.circle(frame, (x_lm, y_lm), 2, (255, 0, 0), -1)

            cv.putText(frame, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0,
                       (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow('Detected face', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

real_time_face_recognition()
