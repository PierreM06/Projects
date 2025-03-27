import mlx.core as mx

activations = {
    'sigmoid': lambda x: 1 / (1 + mx.exp(-x)),
    'sigmoid_derivative': lambda x: x * (1 - x),
    'relu': lambda x: mx.maximum(0, x),
    'relu_derivative': lambda x: (x > 0).astype(mx.array),
    'softmax': lambda x: mx.exp(x - mx.max(x)) / mx.sum(mx.exp(x - mx.max(x)), axis=-1, keepdims=True),
    'softmax_derivative': lambda x, y: x - y
}