from tensorflow.keras.models import load_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
with open("Dikshita.pickle", "rb") as f:
    X, y = pickle.load(f)

# Reshape input data
X = np.array(X, dtype=np.float32).reshape(-1, 42, 2, 1)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load and compile the model
model = load_model("Dikshita.h5")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict classes
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
correct = sum(y_pred == y_test)
incorrect = len(y_test) - correct

# Plot bar chart for correct vs incorrect
plt.figure(figsize=(6, 4))
plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
plt.title(f"Model Prediction Accuracy: {accuracy * 100:.2f}%")
plt.ylabel("Number of Samples")
plt.show()
