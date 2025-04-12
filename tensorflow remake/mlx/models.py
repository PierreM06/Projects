from layers import Layer
from tensor import Tensor
import mlx.core as mx
from losses import Loss


class Model:
    def __init__(self) -> None:
        pass

    def add(self, layer: Layer):
        raise NotImplementedError
    
    def forward(self, input: Tensor):
        raise NotImplementedError
    
    def output(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def outputs(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backwards(self, input: Tensor, target: Tensor):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def train(self, inputs: Tensor, targets: Tensor, epoch: int, validation: tuple[Tensor, Tensor]):
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError

class Sequential(Model):
    def __init__(self, loss: Loss, learning_rate: float=0.01) -> None:
        self.layers: list[Layer] = []
        self.loss = loss
        self.learning_rate: float = learning_rate

        super().__init__()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input: Tensor):
        for i, layer in enumerate(self.layers):
            input = layer.output(input)
    
    def output(self, input: Tensor) -> Tensor:
        self.forward(input=input)
        return self.layers[-1].last_output
    
    def outputs(self, inputs: Tensor) -> Tensor:
        return Tensor([self.output(input) for input in inputs])
    
    def backwards(self, input: Tensor, target: Tensor):
        self.forward(input)

        loss = self.loss(self.layers[-1].last_output, target)
        loss.backward()

        self.update()

    def update(self):
        for i, layer in enumerate(self.layers):
            layer.update()

    def train(self, inputs: Tensor, targets: Tensor, epoch: int, validation: tuple[Tensor, Tensor] | None = None):
        for _ in range(epoch):
            for i in range(len(inputs)):
                self.backwards(inputs[i], targets[i])
        pass

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def __str__(self) -> str:
        return super().__str__()
    