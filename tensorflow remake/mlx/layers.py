from activation import activations
from typing import Callable
import mlx.core as mx


class Layer:
    def __init__(self) -> None:
        self.weights: mx.array = None #type: ignore
        self.biases: mx.array = None #type: ignore

        self.activation: Callable[[mx.array], mx.array] = None #type: ignore
        self.derivative: Callable[[mx.array], mx.array] = None #type: ignore
        self.last_input: mx.array = None #type: ignore
        self.last_output: mx.array = None #type: ignore

        self.dw: mx.array = None #type: ignore
        self.db: mx.array = None #type: ignore

    def output(self, input: mx.array) -> mx.array:
        raise NotImplementedError
    
    def error(self, target: mx.array) -> mx.array:
        raise NotImplementedError
    
    def backwards(self, gradient: mx.array):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, neurons:int, input_size: int, activation: str='sigmoid', learning_rate: float=0.01) -> None:
        super().__init__()
        self.weights = mx.random.normal((neurons, input_size,))
        self.biases = mx.random.normal((neurons,))

        self.activation_name = activation
        self.activation = activations[activation]
        self.derivative = activations[f'{activation}_derivative']
        self.learning_rate = learning_rate

        self.dw = None #type: ignore
        self.db = None #type: ignore

    def weighted_sum(self, input: mx.array) -> mx.array:
        return mx.matmul(self.weights, mx.reshape(input, (-1,1))).squeeze() + self.biases
    
    def output(self, input: mx.array) -> mx.array:
        self.last_input = input
        self.last_output = mx.array(self.activation(self.weighted_sum(input=input)))
        return self.last_output
    
    def backwards(self, gradient: mx.array):
        # grad_input = mx.matmul(self.weights.T, gradient)  # Backpropagated error
        grad_weights = mx.matmul(mx.reshape(gradient, shape=(-1,1)), mx.reshape(self.last_input, shape=(-1,1)).T)
        grad_biases = gradient

        # Update weights and biases
        self.dw = self.learning_rate * grad_weights
        self.db = self.learning_rate * grad_biases
    
    def update(self):
        self.weights -= self.dw
        self.biases -= self.db

    def error(self, target: mx.array) -> mx.array:
        return self.last_output * (1 - self.last_output) * -(target - self.last_output)
    

class HelperLayer(Layer):
    def __init__(self) -> None:
        pass

    def output(self, input: mx.array) -> mx.array:
        raise NotImplementedError
    
    def error(self, target: mx.array) -> mx.array:
        raise NotImplementedError


class Flatten(HelperLayer):
    def __init__(self) -> None:
        super().__init__()

