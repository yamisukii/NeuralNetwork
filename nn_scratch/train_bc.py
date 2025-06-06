import numpy as np
import pandas as pd
import utils
from nn import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data

train_data = pd.read_csv(
    'data/breast-cancer-diagnostic.shuf.lrn.csv'
)

test_data = pd.read_csv(
    'data/breast-cancer-diagnostic.shuf.tes.csv'
)

# Drop ID column and extract features/labels
X = train_data.drop(columns=["ID", "class"])
y = train_data["class"].astype(int)  # convert True/False to 1/0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation split (since test_data has no labels for now)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Define network structure (e.g., 30 input features → 1 hidden layer → 1 output)
nn = NeuralNetwork(
    layer_sizes=[X_train.shape[1], 16, 1],
    activations=[utils.relu, utils.sigmoid],
    learning_rate=0.01,
    multiclass=False
)

# Train
nn.train(X_train, y_train.values, epochs=50)

# Predict and evaluate
preds = nn.predict(X_val)
accuracy = np.mean(preds.flatten() == y_val.values)
print(f"Validation Accuracy: {accuracy:.2%}")
nn.model_size()
