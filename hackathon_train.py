import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load data
with open("hackathon.pickle", "rb") as f:
    X, y = pickle.load(f)

# One-hot encode labels
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Flatten last 2 dims: (30, 42, 2) → (30, 84)
X = X.reshape(-1, 30, 84)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 84)),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Save
model.save("hackathon.h5")
print("✅ Model saved as 'hackathon.h5'")
