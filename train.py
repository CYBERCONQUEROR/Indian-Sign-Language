import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# Load data
with open("Dikshita.pickle", "rb") as f:
    X, y = pickle.load(f)

X = np.array(X, dtype=np.float32).reshape(-1, 42, 2, 1)
y = to_categorical(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3, 2), activation='relu', padding='valid', input_shape=(42, 2, 1)),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(64, (3, 1), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 1)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Save model
model.save("Dikshitaa.h5")
print("✅ Model saved as 'Dikshitaa.h5'")
