from typing import Callable
from tensor import Tensor
import mlx.core as mx


class Layer:
    def __init__(self) -> None:
        self.weights: Tensor = None #type: ignore
        self.biases: Tensor = None #type: ignore

        self.last_input: Tensor = None #type: ignore
        self.last_output: Tensor = None #type: ignore

        self.learning_rate: float = None #type: ignore

    def output(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def parameters(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, neurons:int, input_size: int, activation: str='sigmoid', learning_rate: float=0.01) -> None:
        super().__init__()
        self.weights = Tensor(mx.random.normal((neurons, input_size,)), requires_grad=True)
        self.biases = Tensor(mx.random.normal((neurons,)), requires_grad=True)

        self.activation = activation

    def calculate_weighted_sum(self, input: Tensor) -> Tensor:
        return (self.weights @ input) + self.biases
    
    def output(self, input: Tensor) -> Tensor:
        self.last_input = input
        self.weighted_sum = self.calculate_weighted_sum(input=input)
        self.last_output = self.weighted_sum.apply(self.activation)
        return self.last_output
    
    def update(self):
        self.weights -= self.weights.grad
        self.biases -= self.biases.grad

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.biases]
