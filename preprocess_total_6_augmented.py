import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import random

# === SETTINGS ===
data_dir = '/content/data'
augment = True
target_samples = 84000

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# === Data Containers ===
data = []
labels = []

# === Helper Functions ===
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_value = np.max(np.linalg.norm(centered, axis=1))
    if max_value == 0:
        return centered.flatten().tolist()
    normalized = centered / max_value
    return normalized.flatten().tolist()

def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            return normalize_landmarks(landmarks)
    return None

def augment_image(image):
    aug_images = []

    # Horizontal flip
    aug_images.append(cv2.flip(image, 1))

    # Small rotations
    for angle in [-10, 10]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        aug_images.append(rotated)

    # Small scale
    for scale in [0.9, 1.1]:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        rescaled = cv2.resize(image, (new_w, new_h))

        if scale < 1.0:
            # Add padding
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            padded = cv2.copyMakeBorder(
                rescaled, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
            aug_images.append(padded)
        else:
            # Crop center
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2
            cropped = rescaled[y_start:y_start + h, x_start:x_start + w]
            aug_images.append(cropped)

    return aug_images


# === Processing Images ===
for label in sorted(os.listdir(data_dir)):
    class_dir = os.path.join(data_dir, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"📁 Processing class: {label}")
    image_files = os.listdir(class_dir)
    for img_name in image_files:
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Original
        result = extract_landmarks(img)
        if result:
            data.append(result)
            labels.append(label)

        # Augmented
        if augment:
            for aug_img in augment_image(img):
                result = extract_landmarks(aug_img)
                if result:
                    data.append(result)
                    labels.append(label)

# === Padding if missing due to failed landmark detection ===
while len(data) < target_samples:
    idx = random.randint(0, len(data) - 1)
    data.append(data[idx])
    labels.append(labels[idx])

print(f"\n✅ Total samples: {len(data)} (Expected: 252000)")

# === Save to Pickle ===
with open("after.pickle", "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ after.pickle saved successfully.")
