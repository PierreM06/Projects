from layers import Layer, HelperLayer
import mlx.core as mx


class Model:
    def __init__(self) -> None:
        pass

    def add(self, layer: Layer):
        raise NotImplementedError
    
    def forward(self, input: mx.array):
        raise NotImplementedError
    
    def output(self, input: mx.array) -> mx.array:
        raise NotImplementedError
    
    def outputs(self, inputs: mx.array) -> mx.array:
        raise NotImplementedError
    
    def backwards(self, input: mx.array, target: mx.array):
        raise NotImplementedError
    
    def update(self, deltaW: list[mx.array], deltaB: list[mx.array]):
        raise NotImplementedError
    
    def train(self, inputs: mx.array, targets: mx.array, epoch: int, validation: tuple[mx.array, mx.array]):
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError

class Sequential(Model):
    def __init__(self, learning_rate: float=0.01) -> None:
        self.layers: list[Layer] = []
        self.learning_rate: float = learning_rate

        super().__init__()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input: mx.array):
        for i, layer in enumerate(self.layers):
            layer.output(input if i == 0 else self.layers[i-1].last_output)
    
    def output(self, input: mx.array) -> mx.array:
        self.forward(input=input)
        return self.layers[-1].last_output
    
    def outputs(self, inputs: mx.array) -> mx.array:
        return mx.array([self.output(input) for input in inputs])
    
    def backwards(self, input: mx.array, target: mx.array):
        self.forward(input)

        dz = [self.layers[-1].error(target)]
        dw = [self.learning_rate * mx.reshape(dz[0], shape=(-1,1)) * mx.reshape(input.T if len(self.layers) == 1 else self.layers[-2].last_output.T, shape=(-1,1)).T]
        db = [self.learning_rate * dz[0]]

        for i in range(len(self.layers)-1):
            dz.append(self.layers[-2-i].derivative(self.layers[-2-i].last_output) * mx.matmul(self.layers[-1-i].weights.T, dz[i]))
            dw.append(self.learning_rate * mx.matmul(mx.reshape(dz[1+i], (-1,1)), mx.reshape((self.layers[-3-i].last_output if abs(-3-i) < len(self.layers) else input), (-1,1)).T))
            db.append(self.learning_rate * dz[1+i])

        dw.reverse()
        db.reverse()
        self.update(dw, db)

    def update(self, deltaW: list[mx.array], deltaB: list[mx.array]):
        for i, layer in enumerate(self.layers):
            layer.update(deltaW=deltaW[i], deltaB=deltaB[i])

    def train(self, inputs: mx.array, targets: mx.array, epoch: int, validation: tuple[mx.array, mx.array] | None = None):
        for _ in range(epoch):
            for i in range(len(inputs)):
                self.backwards(inputs[i], targets[i])
    
    def __str__(self) -> str:
        return super().__str__()
    