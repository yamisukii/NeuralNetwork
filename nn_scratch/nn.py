import numpy as np
import utils


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, multiclass=False):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.multiclass = multiclass
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
        return params

    def forward(self, X):
        self.cache = {}
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
        grads = {}
        m = X.shape[0]
        L = len(self.layer_sizes) - 1

        # Output layer
        A_final = self.cache[f"A{L}"]
        # Output layer gradient

        if self.multiclass:
            dZ = A_final - y  # y = one-hot
        else:
            dZ = A_final - y.reshape(-1, 1)  # y = scalar, reshape to column

        A_prev = self.cache[f"A{L-1}"] if L > 1 else X
        grads[f"dW{L}"] = np.dot(A_prev.T, dZ) / m
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

        # Hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.params[f"W{l+1}"].T)
            Z = self.cache[f"Z{l}"]
            A = self.cache[f"A{l}"]

            activation_fn = self.activations[l - 1]
            if activation_fn == utils.relu:
                dZ = dA * utils.relu_derivative(Z)
            elif activation_fn == utils.sigmoid:
                dZ = dA * utils.sigmoid_derivative(Z)
            elif activation_fn == utils.tanh:
                dZ = dA * utils.tanh_derivative(A)

            A_prev = self.cache[f"A{l-1}"] if l > 1 else X
            grads[f"dW{l}"] = np.dot(A_prev.T, dZ) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    def update_weights(self, grads):
        for l in range(1, len(self.layer_sizes)):
            self.params[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_loss = 0

            for i in range(X.shape[0]):
                # single example, shape (1, input_size)
                xi = X[i].reshape(1, -1)
                yi = y[i]  # label (scalar or shape (1,))

                # Forward pass
                output = self.forward(xi)

                # Compute loss (scalar)
                # Im Training pro Beispiel:
                if self.multiclass:
                    loss = utils.categorical_cross_entropy_single(
                        output.flatten(), yi)
                else:
                    loss = utils.binary_cross_entropy_single(output, yi)

                total_loss += loss

                # Backward pass
                grads = self.backward(xi, yi, output)

                # Update weights using gradients
                self.update_weights(grads)

            avg_loss = total_loss / X.shape[0]
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss.item():.4f}")

    def predict(self, X):
        outputs = self.forward(X)  # shape: (n_samples, 1)
        if self.multiclass:
            return np.argmax(outputs, axis=1)  # class index
        else:
            return (outputs >= 0.5).astype(int)  # binary 0/1

    def model_size(self, verbose=True):
        total_params = 0
        total_bytes = 0

        for l in range(1, len(self.layer_sizes)):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            w_params = W.size
            b_params = b.size
            layer_params = w_params + b_params

            total_params += layer_params
            # assuming float64 (8 bytes per param)
            total_bytes += layer_params * 8

            if verbose:
                print(
                    f"Layer {l}: W{l}.shape = {W.shape}, b{l}.shape = {b.shape}, params = {layer_params}")

        if verbose:
            print(f"\nTotal parameters: {total_params}")
            print(
                f"Estimated RAM usage: {total_bytes / 1024:.2f} KB ({total_bytes / (1024 ** 2):.2f} MB)")

        return total_params, total_bytes
