import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load full iris dataset
data = load_iris()
X = data.data
y = data.target

# Only keep classes 0 and 1 (Setosa and Versicolor)
binary_mask = (y == 0) | (y == 1)
X = X[binary_mask]
y = y[binary_mask]  # already 0 or 1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import utils  # Your utils.py should already be defined
from nn import NeuralNetwork  # Your class with forward/backward/train/etc.

# Define network: 4 inputs → 1 hidden layer (e.g. 5 nodes) → 1 output (sigmoid)
nn = NeuralNetwork(
    layer_sizes=[4, 5, 1], activations=[utils.relu, utils.sigmoid], learning_rate=0.1
)

# Train the network
nn.train(X_train, y_train, epochs=100)

# Predict on test data
predictions = nn.predict(X_test)
print("Predictions:", predictions.flatten())
print("True Labels:", y_test)
