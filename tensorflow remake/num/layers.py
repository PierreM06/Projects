from activation import activations
import numpy as np


class Layer:
    def __init__(self) -> None:
        self.weights: np.ndarray = None #type: ignore
        self.biases: np.ndarray = None #type: ignore

        self.activation: Callable[[np.ndarray], np.ndarray] = None #type: ignore
        self.derivative: Callable[[np.ndarray], np.ndarray] = None #type: ignore
        self.last_output: np.ndarray = None #type: ignore

    def output(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def error(self, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def update(self, deltaW: np.ndarray, deltaB: np.ndarray):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, neurons:int, input_size: int, activation: str='sigmoid') -> None:
        super().__init__()
        self.weights = np.random.normal(size=(neurons, input_size,))
        self.biases = np.random.normal(size=(neurons,))

        self.activation_name = activation
        self.activation = activations[activation]
        self.derivative = activations[f'{activation}_derivative']

    def weighted_sum(self, input: np.ndarray) -> np.ndarray:
        return np.dot(self.weights, input) + self.biases
    
    def output(self, input: np.ndarray) -> np.ndarray:
        self.last_output = np.array(self.activation(self.weighted_sum(input=input)))
        return self.last_output
    
    def update(self, deltaW: np.ndarray, deltaB: np.ndarray):
        self.weights -= deltaW
        self.biases -= deltaB

    def error(self, target: np.ndarray) -> np.ndarray:
        return self.last_output * (1-self.last_output) * -(target - self.last_output)
