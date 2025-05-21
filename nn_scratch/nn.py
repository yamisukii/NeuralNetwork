import numpy as np
import utils


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = self._init_weights()
        self.activations = activations

    def _init_weights(self):
        params = {}
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            # generating random weights
            W = utils.generate_wt(input_size, output_size)
            b = np.zeros((1, output_size))

            params[f"W{i+1}"] = W
            params[f"b{i+1}"] = b

    def forward(self, X):
        self.cache = {}  # store z and a values for backprop
        a = X

        for i in range(1, len(self.layer_sizes)):
            w = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            z = np.dot(a, w) + b
            self.cache[f"Z{i}"] = z

            activation_fn = self.activations[i - 1]
            a = activation_fn(z)
            self.cache[f"A{i}"] = a

        return a

    def backward(self, X, y, outputs):
        # Compute gradients
        ...

    def update_weights(self, grads):
        ...

    def train(self, X, y, epochs):
        ...

    def predict(self, X):
        ...
