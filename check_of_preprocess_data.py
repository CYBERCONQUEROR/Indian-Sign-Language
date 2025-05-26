import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load your pickle file
pickle_path = r"Dikshita.pickle"
with open(pickle_path, 'rb') as f:
    dataset = pickle.load(f)

# Check the structure of the loaded dataset
print(type(dataset))  # Should print <class 'tuple'> or <class 'dict'>
print(dataset)        # Inspect the dataset to understand its structure

# If dataset is a tuple, access its elements
if isinstance(dataset, tuple):
    data = dataset[0]
    labels = dataset[1]
else:
    # If it's a dictionary (for example), use the original code
    data = dataset['data']
    labels = dataset['labels']

# Print basic information
print("✅ Total Samples:", len(labels))

