import cv2
import mediapipe as mp
import numpy as np
import math
import xgboost as xgb
import pandas as pd
import pyttsx3

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 145)
tts_engine.setProperty('volume', 1.0)

# Load the model
model_file_path = './xgboost_asl_model_scoring.json'
model = xgb.Booster()
model.load_model(model_file_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.4)

# Helper functions
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2, x3, y3):
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

labels_dict = {i: chr(65 + i) for i in range(26)}

sentence = ""
frame_count = 0
char_counter = {}
frame_buffer = 25  # Frames to confirm a character
space_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_detected = False
    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            x_ = [(x - min_x) / (max_x - min_x) if max_x != min_x else 0 for x in x_]
            y_ = [(y - min_y) / (max_y - min_y) if max_y != min_y else 0 for y in y_]

            distances = [calculate_distance(x_[0], y_[0], x_[i], y_[i]) for i in range(1, len(x_))]
            angles = [calculate_angle(x_[0], y_[0], x_[i], y_[i], x_[i + 1], y_[i + 1]) for i in range(1, len(x_) - 1)]

            if len(x_) == 21 and len(y_) == 21 and len(distances) >= 20 and len(angles) >= 19:
                feature_dict = {f'x_{i}': x_[i] for i in range(21)}
                feature_dict.update({f'y_{i}': y_[i] for i in range(21)})
                feature_dict.update({f'dist_{i}': distances[i] for i in range(20)})
                feature_dict.update({f'angle_{i}': angles[i] for i in range(19)})

                try:
                    dmatrix = xgb.DMatrix(pd.DataFrame([feature_dict]))
                    prediction = model.predict(dmatrix)
                    predicted_index = int(prediction[0])
                    predicted_character = labels_dict.get(predicted_index, "?")

                    if predicted_character not in char_counter:
                        char_counter[predicted_character] = 1
                    else:
                        char_counter[predicted_character] += 1

                    if char_counter[predicted_character] >= frame_buffer:
                        sentence += predicted_character
                        tts_engine.say(predicted_character)
                        tts_engine.runAndWait()
                        char_counter.clear()

                    cv2.putText(frame, f"Prediction: {predicted_character}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Prediction error: {e}")

    frame_count += 1
    if not hand_detected and frame_count % 15 == 0:
        if not space_detected:
            sentence += " "
            space_detected = True
    elif hand_detected:
        space_detected = False

    (text_width, text_height), _ = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10 - 5, 100 - text_height - 5), (10 + text_width + 5, 100 + 5), (0, 0, 0), -1)
    cv2.putText(frame, sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if sentence.strip():
            print("Final Sentence:", sentence)
            tts_engine.say("Final sentence: " + sentence)
            tts_engine.runAndWait()
        break

cap.release()
cv2.destroyAllWindows()
