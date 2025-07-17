import os
import cv2

# Set the data directory path where the dataset will be saved
DATA_DIR = './data'

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (A-Z)
number_of_classes = 26
# Number of images to capture per class
dataset_size = 200

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Create subdirectories for each class (A-Z)
for j in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, chr(65 + j))  # 'A' to 'Z'
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    print(f'Collecting data for class {chr(65 + j)}')

    done = False
    while not done:
        ret, frame = cap.read()
        # Display prompt on the screen
        cv2.putText(frame, 'Ready? Press "Q" to start capturing!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # Wait for user to press 'Q' to start capturing
        if cv2.waitKey(25) == ord('q'):
            done = True

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the image to the corresponding class folder
        cv2.imwrite(os.path.join(class_folder, f'{counter + 1}.jpg'), frame)
        counter += 1

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
