import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

data_dir = r"C:\Users\Krishnav\OneDrive\Desktop\AKTU\images"  # Your dataset path
IMAGE_SIZE = 256  # Fixed input size for model

X = []
y = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Resize + pad to square
def resize_and_pad(img, size=IMAGE_SIZE):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    delta_w, delta_h = size - new_w, size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

# Normalize 42 landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 2)
    base = landmarks[0]
    centered = landmarks - base
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_dist).flatten().tolist() if max_dist != 0 else centered.flatten().tolist()

# Extract 42 (2 hands × 21) normalized landmarks
def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            lms1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            lms2 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[1].landmark]
        else:
            lms1 = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            lms2 = [[0, 0]] * 21
        combined = lms1 + lms2
        return normalize_landmarks(combined)
    return None

# Augmentation: flip, rotate, scale
def augment_image(image):
    aug_images = []
    aug_images.append(cv2.flip(image, 1))  # Horizontal flip

    for angle in [-10, 10]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug_images.append(cv2.warpAffine(image, M, (w, h)))

    for scale in [0.9, 1.1]:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        if scale < 1.0:
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            padded = cv2.copyMakeBorder(resized, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                                        borderType=cv2.BORDER_CONSTANT, value=0)
            aug_images.append(padded)
        else:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            cropped = resized[start_y:start_y + h, start_x:start_x + w]
            aug_images.append(cropped)

    return aug_images  # Total: 6 augmentations

# === Load and process dataset ===
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

        img_resized = resize_and_pad(img)

        # Original image
        result = extract_landmarks(img_resized)
        if result:
            X.append(result)
            y.append(label)

        # Augmented images
        for aug_img in augment_image(img_resized):
            result = extract_landmarks(aug_img)
            if result:
                X.append(result)
                y.append(label)

hands.close()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Final dataset shape
X = np.array(X, dtype=np.float32).reshape(-1, 42, 2, 1)  # For CNN
y_encoded = np.array(y_encoded)

# Save
with open("Dikshita.pickle", "wb") as f:
    pickle.dump((X, y_encoded), f)

with open("dikshita.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ Preprocessing complete. Total samples: {len(X)}")
print("✅ Saved as 'double_hand_data.pickle' and label encoder")
