import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = self._init_weights()
        self.activations = activations

    def _init_weights(self):
        # Initialize weights and biases
        ...

    def forward(self, X):
        # Compute layer outputs
        ...

    def backward(self, X, y, outputs):
        # Compute gradients
        ...

    def update_weights(self, grads):
        ...

    def train(self, X, y, epochs):
        ...

    def predict(self, X):
        ...
