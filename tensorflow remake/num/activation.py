import numpy as np

activations = {
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'sigmoid_derivative': lambda x: x * (1 - x),
    'relu': lambda x: np.maximum(0, x),
    'relu_derivative': lambda x: (x > 0).astype(float),
    'softmax': lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) / 
                         np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True),
    'softmax_derivative': lambda x, y: x - y
}