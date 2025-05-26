import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import pickle

# Load model and label encoder
model = load_model("sign_language_cnn_model2.h5")
with open(r"F:\ISL\all data\label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Normalize function (same as training)
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    if max_value == 0:
        return centered
    normalized = centered / max_value
    return normalized

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            normalized = normalize_landmarks(landmarks)

            # Reshape to match CNN input: (21, 2, 1)
            input_data = np.array(normalized, dtype=np.float32).reshape(1, 21, 2, 1)

            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_class])[0]

            # Display
            cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
