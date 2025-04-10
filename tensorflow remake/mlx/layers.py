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

        self.dw: Tensor = None #type: ignore
        self.db: Tensor = None #type: ignore

    def output(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backwards(self, gradient: Tensor):
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

        self.dw = None #type: ignore
        self.db = None #type: ignore

    def calculate_weighted_sum(self, input: Tensor) -> Tensor:
        return (self.weights @ input) + self.biases
    
    def output(self, input: Tensor) -> Tensor:
        self.last_input = input
        self.weighted_sum = self.calculate_weighted_sum(input=input)
        self.last_output = self.weighted_sum.apply(self.activation)
        return self.last_output
    
    def backwards(self, gradient: Tensor) -> None:
        # # grad_input = mx.matmul(self.weights.T, gradient)  # Backpropagated error
        # grad_weights = mx.matmul(mx.reshape(gradient, shape=(-1,1)), mx.reshape(self.last_input, shape=(-1,1)).T)
        # grad_biases = gradient

        # # Update weights and biases
        # self.dw = grad_weights * self.learning_rate
        # self.db = grad_biases * self.learning_rate
        pass
    
    def update(self):
        self.weights -= self.dw
        self.biases -= self.db

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.biases]
