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

        # === Output layer ===
        A_final = self.cache[f"A{L}"]
        dZ = A_final - y if self.multiclass else (A_final - y.reshape(-1, 1))
        A_prev = self.cache.get(f"A{L-1}", X)

        grads[f"dW{L}"] = np.dot(A_prev.T, dZ) / m
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

        # === Hidden layers ===
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.params[f"W{l+1}"].T)
            Z = self.cache[f"Z{l}"]
            A = self.cache[f"A{l}"]
            A_prev = self.cache.get(f"A{l-1}", X)

            act_fn = self.activations[l - 1]
            act_name = act_fn.__name__

            dZ = dA * utils.activation_derivative(act_name, Z, A)

            grads[f"dW{l}"] = np.dot(A_prev.T, dZ) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    def update_weights(self, grads):
        for l in range(1, len(self.layer_sizes)):
            self.params[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def train(self, X, y, epochs, X_val=None, y_val=None):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for i in range(X.shape[0]):
                xi = X[i].reshape(1, -1)
                yi = y[i].reshape(
                    1, -1) if self.multiclass else np.array([[y[i]]])

                output = self.forward(xi)

                # Loss + prediction
                if self.multiclass:
                    loss = utils.categorical_cross_entropy(
                        output.flatten(), yi.flatten())
                    pred = np.argmax(output)
                    label = np.argmax(yi)
                else:
                    loss = utils.binary_cross_entropy_single(output, yi)
                    pred = int(output >= 0.5)
                    label = int(yi[0][0])

                total_loss += loss
                correct += int(pred == label)

                # Backprop + update
                grads = self.backward(xi, yi, output)
                self.update_weights(grads)

            # Epoch stats
            avg_loss = total_loss / X.shape[0]
            acc = correct / X.shape[0]
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(acc)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate_on_validation(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {float(avg_loss):.4f}, Acc: {float(acc):.4f}, "
                    f"Val Loss: {float(val_loss):.4f}, Val Acc: {float(val_acc):.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {float(avg_loss):.4f}, Acc: {float(acc):.4f}")

    def predict(self, X):
        outputs = self.forward(X)
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
            total_bytes += layer_params * 8

            if verbose:
                print(
                    f"Layer {l}: W{l}.shape = {W.shape}, b{l}.shape = {b.shape}, params = {layer_params}"
                )

        if verbose:
            print(f"\nTotal parameters: {total_params}")
            print(
                f"Estimated RAM usage: {total_bytes / 1024:.2f} KB ({total_bytes / (1024 ** 2):.2f} MB)"
            )

        return total_params, total_bytes

    def evaluate_on_validation(self, X_val, y_val):
        preds = self.predict(X_val)

        if self.multiclass:
            val_loss = np.mean([
                utils.categorical_cross_entropy(
                    self.forward(X_val[i].reshape(1, -1)).flatten(),
                    y_val[i].flatten()
                )
                for i in range(X_val.shape[0])
            ])
            true_labels = np.argmax(y_val, axis=1)
        else:
            val_loss = np.mean([
                utils.binary_cross_entropy_single(
                    self.forward(X_val[i].reshape(1, -1)),
                    np.array([[y_val[i]]])
                )
                for i in range(X_val.shape[0])
            ])
            true_labels = y_val

        acc = np.mean(preds.flatten() == true_labels)
        return float(val_loss), float(acc)
