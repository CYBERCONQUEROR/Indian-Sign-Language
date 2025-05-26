import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATA_DIR = r"C:\Users\Krishnav\OneDrive\Desktop\AKTU\images"  # <-- update this
SEQUENCE_LENGTH = 30

X = []
y = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Normalize 42 landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_dist) if max_dist != 0 else centered

# Extract 42 normalized keypoints (21 per hand × 2)
def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hands_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hands_landmarks.extend([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        if len(hands_landmarks) < 42:
            hands_landmarks += [[0, 0]] * (42 - len(hands_landmarks))
        return normalize_landmarks(hands_landmarks)
    return None

# Process each folder (label)
for label in tqdm(os.listdir(DATA_DIR), desc="Processing Labels"):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    images = sorted(os.listdir(label_dir))
    sequences = [images[i:i+SEQUENCE_LENGTH] for i in range(0, len(images)-SEQUENCE_LENGTH+1, SEQUENCE_LENGTH)]

    for seq in sequences:
        frame_sequence = []
        for img_file in seq:
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            lm = extract_landmarks(img)
            if lm is not None:
                frame_sequence.append(lm)
        
        if len(frame_sequence) == SEQUENCE_LENGTH:
            X.append(frame_sequence)
            y.append(label)

hands.close()

X = np.array(X, dtype=np.float32)   # Shape: (samples, 30, 42, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save
with open("hackathon.pickle", "wb") as f:
    pickle.dump((X, y_encoded), f)

with open("hackathon_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ Saved {len(X)} sequences.")
print("✅ Shapes → X:", X.shape, " y:", len(y_encoded))
