from layers import Layer
import numpy as np

class Model:
    def __init__(self) -> None:
        pass

    def add(self, layer: Layer):
        raise NotImplementedError
    
    def forward(self, input: np.ndarray):
        raise NotImplementedError
    
    def output(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def outputs(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backwards(self, input: np.ndarray, target: np.ndarray):
        raise NotImplementedError
    
    def update(self, deltaW: list[np.ndarray], deltaB: list[np.ndarray]):
        raise NotImplementedError
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, epoch: int, validation: tuple[np.ndarray, np.ndarray]):
        raise NotImplementedError
    
    def __str__(self) -> str:
        return super().__str__()

class Sequential(Model):
    def __init__(self, learning_rate: float=0.01) -> None:
        self.layers: list[Layer] = []
        self.learning_rate: float = learning_rate
        super().__init__()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input: np.ndarray):
        for i, layer in enumerate(self.layers):
            layer.output(input if i == 0 else self.layers[i-1].last_output)
    
    def output(self, input: np.ndarray) -> np.ndarray:
        self.forward(input=input)
        return self.layers[-1].last_output
    
    def outputs(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([self.output(input) for input in inputs])
    
    def backwards(self, input: np.ndarray, target: np.ndarray):
        self.forward(input)

        dz = [self.layers[-1].error(target)]
        dw = [self.learning_rate * dz[0] * input if len(self.layers) == 1 else self.layers[-2].last_output]
        db = [self.learning_rate * dz[0]]

        for i in range(len(self.layers)-1):
            dz.append(self.layers[-2-i].derivative(self.layers[-2-i].last_output) * np.dot(self.layers[-1-i].weights.T, dz[i]))
            dw.append(self.learning_rate * np.dot(dz[1+i], self.layers[-2-i].last_output))
            db.append(self.learning_rate * dz[1+i])

        dw.reverse()
        db.reverse()
        self.update(dw, db)

    def update(self, deltaW: list[np.ndarray], deltaB: list[np.ndarray]):
        for i, layer in enumerate(self.layers):
            layer.update(deltaW=deltaW[i], deltaB=deltaB[i])

    def train(self, inputs: np.ndarray, targets: np.ndarray, epoch: int, validation: tuple[np.ndarray, np.ndarray] | None = None):
        for _ in range(epoch):
            for i in range(len(inputs)):
                self.backwards(inputs[i], targets[i])

    def __str__(self) -> str:
        return super().__str__()
    