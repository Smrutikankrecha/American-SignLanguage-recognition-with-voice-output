import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

# Define data directory
DATA_DIR = './data'
CSV_FILE = 'asl_landmarks_800.csv'


def calculate_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(x1, y1, x2, y2, x3, y3):
    """Calculates the angle (in degrees) between three points."""
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def preprocess_image(img_path):
    """Extracts and normalizes hand landmarks, distances, and angles."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None  # Skip invalid image

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize coordinates
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            x_ = [(x - min_x) / (max_x - min_x) if max_x != min_x else 0 for x in x_]
            y_ = [(y - min_y) / (max_y - min_y) if max_y != min_y else 0 for y in y_]

            # Compute distances (e.g., from wrist to other key points)
            distances = [calculate_distance(x_[0], y_[0], x_[i], y_[i]) for i in range(1, len(x_))]

            # Compute angles (e.g., between joints)
            angles = [calculate_angle(x_[0], y_[0], x_[i], y_[i], x_[i + 1], y_[i + 1]) for i in range(1, len(x_) - 1)]

            return x_ + y_ + distances + angles
    else:
        print(f"No hands detected in {img_path}")
        return None


def create_csv_file(data_dir, csv_filename):
    """Creates a CSV file containing processed hand landmark data."""
    header = [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)] + \
             [f"dist_{i}" for i in range(20)] + [f"angle_{i}" for i in range(19)] + ["label"]

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for dir_ in os.listdir(data_dir):
            class_path = os.path.join(data_dir, dir_)
            if not os.path.isdir(class_path):
                continue
            for img_path in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_path)
                processed_data = preprocess_image(img_full_path)
                if processed_data:
                    writer.writerow(processed_data + [dir_])

    print(f"CSV file {csv_filename} created successfully.")


if __name__ == "__main__":
    create_csv_file(DATA_DIR, CSV_FILE)
