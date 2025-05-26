import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import random

data_dir = r"C:\Users\Krishnav\Downloads\archive\data"  # your dataset folder
X = []
y = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_value).flatten().tolist() if max_value != 0 else centered.flatten().tolist()

def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            landmarks1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            landmarks2 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[1].landmark]
        else:
            landmarks1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            landmarks2 = [[0, 0]] * 21
        combined = landmarks1 + landmarks2  # (42, 2)
        return normalize_landmarks(combined)
    return None

def augment_image(image):
    aug_images = []

    # Horizontal flip
    aug_images.append(cv2.flip(image, 1))

    # Small rotations
    for angle in [-10, 10]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug_images.append(cv2.warpAffine(image, M, (w, h)))

    # Scaling
    for scale in [0.9, 1.1]:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        if scale < 1.0:
            # Padding
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            padded = cv2.copyMakeBorder(
                resized, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
            aug_images.append(padded)
        else:
            # Cropping
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            cropped = resized[start_y:start_y + h, start_x:start_x + w]
            aug_images.append(cropped)

    return aug_images  # Total: 6 augmentations

# Data loading and processing
for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"📁 Processing class: {label}")
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Original image
        result = extract_landmarks(img)
        if result:
            X.append(result)
            y.append(label)

        # Augmented images
        for aug_img in augment_image(img):
            result = extract_landmarks(aug_img)
            if result:
                X.append(result)
                y.append(label)

hands.close()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Final shaping
X = np.array(X, dtype=np.float32).reshape(-1, 42, 2, 1)
y = np.array(y)

# Save dataset
with open("double_hand_data.pickle", "wb") as f:
    pickle.dump((X, y_encoded), f)

with open("double_hand_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ Preprocessing complete. Total samples: {len(X)}")
print("✅ Data saved to 'double_hand_data.pickle'")
