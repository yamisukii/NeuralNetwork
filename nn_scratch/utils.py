import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return (1/(1 + np.exp(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def generate_wt(x, y):
    li = []
    for i in range(x * y):
        li.append(np.random.randn())
    return (np.array(li).reshape(x, y))
