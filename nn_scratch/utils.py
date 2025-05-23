import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(a):
    return 1 - a ** 2


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def generate_wt(x, y):
    li = []
    for i in range(x * y):
        li.append(np.random.randn())
    return np.array(li).reshape(x, y)


def binary_cross_entropy_single(y_hat, y):
    """
    y_hat: predicted probability (scalar, after sigmoid)
    y: true label (0 or 1)
    """
    epsilon = 1e-12  # to avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss


def categorical_cross_entropy(y_hat, y):
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.sum(y * np.log(y_hat))


def tanh(x):
    return np.tanh(x)
