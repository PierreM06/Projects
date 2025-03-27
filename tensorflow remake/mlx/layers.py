from activation import activations
from typing import Callable
import mlx.core as mx


class Layer:
    def __init__(self) -> None:
        self.weights: mx.array = None #type: ignore
        self.biases: mx.array = None #type: ignore

        self.activation: Callable[[mx.array], mx.array] = None #type: ignore
        self.derivative: Callable[[mx.array], mx.array] = None #type: ignore
        self.last_output: mx.array = None #type: ignore

    def output(self, input: mx.array) -> mx.array:
        raise NotImplementedError
    
    def error(self, target: mx.array) -> mx.array:
        raise NotImplementedError
    
    def backwards(self):
        raise NotImplementedError
    
    def update(self, deltaW: mx.array, deltaB: mx.array):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, neurons:int, input_size: int, activation: str='sigmoid') -> None:
        super().__init__()
        self.weights = mx.random.normal((neurons, input_size,))
        self.biases = mx.random.normal((neurons,))

        self.activation_name = activation
        self.activation = activations[activation]
        self.derivative = activations[f'{activation}_derivative']

    def weighted_sum(self, input: mx.array) -> mx.array:
        return mx.matmul(self.weights, mx.reshape(input, (-1,1))).squeeze() + self.biases
    
    def output(self, input: mx.array) -> mx.array:
        self.last_output = mx.array(self.activation(self.weighted_sum(input=input)))
        return self.last_output
    
    def update(self, deltaW: mx.array, deltaB: mx.array):
        self.weights -= deltaW
        self.biases -= deltaB

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

