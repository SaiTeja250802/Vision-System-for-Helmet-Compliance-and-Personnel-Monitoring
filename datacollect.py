import os
import cv2 as cv

# Directory to save the dataset
dataset_path = r'D:\learn\ML\OpenCV\dataset'


def capture_images(person_name, num_images=50):
    person_path = os.path.join(dataset_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv.imshow('Frame', frame)
        img_path = os.path.join(person_path, f'img{count + 1}.jpg')
        cv.imwrite(img_path, frame)
        count += 1
        print(f'Image {count}/{num_images} captured')

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


while True:
    person_name = input("Enter the name of the person (or type 'exit' to stop): ").strip()
    if person_name.lower() == 'exit':
        break
    num_images = int(input("Enter the number of images to capture: ").strip())

    print(f'Capturing images for {person_name}')
    capture_images(person_name, num_images=num_images)
    print(f'Finished capturing images for {person_name}')
