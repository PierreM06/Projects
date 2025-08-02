from typing import Callable, Any
from .tensor import Tensor
import mlx.core as mx
from abc import ABC, ABCMeta, abstractmethod

LAYER_REGISTRY = {}

class LayerMeta(ABCMeta):  # <-- Subclass ABCMeta instead of type
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Avoid registering the abstract base class itself
        if not cls.__abstractmethods__:
            LAYER_REGISTRY[cls.__name__] = cls


def resolve_initializer(initializer: Callable | str | list | tuple, shape: tuple, value: complex=1) -> mx.array:
    """
    Resolves the initializer to a Tensor based on the provided shape.
    
    :param initializer: Can be a callable, string, list, or tuple.
    :param shape: The shape of the tensor to be initialized.
    :return: A Tensor initialized according to the specified initializer.
    """
    if isinstance(initializer, mx.array):
        return initializer
    if isinstance(initializer, (list, tuple)):
        return mx.array(initializer)
    if callable(initializer):
        return initializer(shape)
    if isinstance(initializer, str):
        if initializer == "xavier":
            fan_in, fan_out = shape[0], shape[1]
            scale = mx.sqrt(2.0 / (fan_in + fan_out))
            return mx.random.normal(shape=shape) * scale
        elif initializer == "he":
            fan_in = shape[0]
            scale = mx.sqrt(2.0 / fan_in)
            return mx.random.normal(shape=shape) * scale
        elif initializer == "normal":
            return mx.random.normal(shape=shape)
        elif initializer == "uniform":
            return mx.random.uniform(low=-1.0, high=1.0, shape=shape)
        elif initializer == "zeros":
            return mx.zeros(shape)
        elif initializer == "ones":
            return mx.ones(shape)
        elif initializer == "constant":
            return mx.full(shape, value)
        else:
            raise ValueError(f"Unknown initializer string: {initializer}")
    raise TypeError(f"Invalid initializer type: {type(initializer)}")

def im2col(x, kernel_size, stride, padding):
    N, H, W, C = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding

    if PH > 0 or PW > 0:
        x = x.pad(((0, 0), (PH, PH), (PW, PW), (0, 0)))  # pad height/width only

    H_padded, W_padded = x.shape[1], x.shape[2]
    H_out = (H_padded - KH) // SH + 1
    W_out = (W_padded - KW) // SW + 1

    patches = []
    for i in range(H_out):
        for j in range(W_out):
            h = i * SH
            w = j * SW
            patch = x[:, h:h+KH, w:w+KW, :]  # shape: (N, KH, KW, C)
            patches.append(patch.reshape((N, -1)))  # flatten to (N, KH * KW * C)

    return Tensor.stack(patches, axis=1)  # shape: (N, H_out * W_out, KH * KW * C)


class Layer(ABC, metaclass=LayerMeta):
    def __init__(self, name: str='', trainable: bool=True):
        super().__init__()
        self._parameters: list[Tensor] = []
        self._buffers: list[Tensor] = []
        self._name: str = name if name else self.__class__.__name__

        self.last_input: Tensor = None  #type: ignore
        self.last_output: Tensor = None  #type: ignore

        self.training: bool = trainable

        self.input_shape = []
        self.output_shape = []

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def set_training(self, mode: bool):
        self.training = mode

    def add_parameter(self, tensor: Tensor, trainable: bool = True) -> Tensor:
        tensor.requires_grad = trainable
        self._parameters.append(tensor)
        return tensor

    def add_buffer(self, tensor: Tensor):
        # Used for non-trainable state (e.g., running stats in BatchNorm)
        tensor.requires_grad = False
        self._buffers.append(tensor)
        return tensor

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def parameters(self) -> list[Tensor]:
        return [p for p in self._parameters if p.requires_grad]

    def all_tensors(self) -> list[Tensor]:
        # Parameters + buffers
        return self._parameters + self._buffers
    
    def get_config(self) -> dict[str, Any]:
        config = {
            "name": self._name,
            "class_name": self.__class__.__name__,
        }
        return config
    
    @classmethod
    def from_config(cls, config: dict[str, Any]):
        config.pop("class_name")
        return cls(**config)
    
    def get_weights(self) -> dict[str, mx.array]:
        # Return dict of all params: weights, biases, gamma, beta, etc.
        # For layers without params, return empty dict.
        return {}

    def set_weights(self, weights: dict[str, mx.array]):
        # Assign weights from dict to layer params
        pass
    

class Activation(Layer):
    def __init__(self, activation: str, name: str=''):
        super().__init__(name)
        assert hasattr(Tensor, activation), f"Unknown activation: {activation}"
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x
        self.last_output = getattr(x, self.activation)()
        return self.last_output
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"activation": self.activation})
        return config


class Dense(Layer):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 weight_init: Callable | str | list | tuple = 'xavier',
                 bias_init: Callable | str | list | tuple = 'zeros',
                 activation: str= '',
                 name: str=''
                 ):
        super().__init__(name)

        self.weights = self.add_parameter(
            Tensor(resolve_initializer(weight_init, (in_features, out_features)), name='weights'),
        )

        self.biases = self.add_parameter(
            Tensor(resolve_initializer(bias_init, (out_features,)), name='biases')
        )

        self.weighted_sum = None
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x
        self.last_output = x @ self.weights + self.biases
        if self.activation:
            self.last_output = getattr(self.last_output, self.activation)()
        return self.last_output
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_features": self.weights.shape[0],
            "out_features": self.weights.shape[1],
            "activation": self.activation
        })
        return config
    
    def get_weights(self):
        return {
            "weights": self.weights.data,
            "biases": self.biases.data,
        }

    def set_weights(self, weights):
        self.weights.data = weights["weights"]
        self.biases.data = weights["biases"]


class Flatten(Layer):
    def __init__(self, name: str=''):
        super().__init__(name)
        self.last_shape = None

    def forward(self, x: Tensor) -> Tensor:
        self.last_shape = x.data.shape
        batch_size = x.data.shape[0]
        flattened = x.reshape((batch_size, -1))
        return flattened
    
    def parameters(self) -> list:
        return []  # Flatten has no parameters


class Dropout(Layer):
    def __init__(self, p: float = 0.5, name: str=''):
        super().__init__(name)
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Generate dropout mask
            keep_prob = 1.0 - self.p
            self.mask = mx.random.bernoulli(keep_prob, shape=x.data.shape) / keep_prob
            out = x * self.mask
            return out
        else:
            return x
        
    def parameters(self) -> list:
        return []  # Dropout has no parameters
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "p": self.p
        })
        return config


class BatchNorm(Layer):
    def __init__(self, 
                 dim: int, 
                 momentum: float = 0.9, 
                 epsilon: float = 1e-5, 
                 trainable: bool = True,
                 name: str=''):
        super().__init__(name=name, trainable=trainable)

        self.gamma: Tensor = self.add_parameter(
            Tensor(mx.ones((1, dim)), name='gamma'),
            trainable
        )

        self.beta: Tensor = self.add_parameter(
            Tensor(mx.zeros((1, dim)), name='beta'),
            trainable
        )

        self.momentum = momentum
        self.epsilon = epsilon

        self.running_mean = Tensor(mx.zeros((1, dim)), name='running_mean')
        self.running_var = Tensor(mx.ones((1, dim)), name='running_var')

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x

        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            batch_var = ((x - batch_mean) ** 2).mean(axis=0, keepdims=True)

            normalized = (x - batch_mean) / (batch_var + self.epsilon) ** 0.5
            out = self.gamma * normalized + self.beta

            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            normalized = (x - self.running_mean) / (self.running_var + self.epsilon) ** 0.5
            out = self.gamma * normalized + self.beta

        self.last_output = out
        return out
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.gamma.shape[1],
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "trainable": self.training
        })
        return config
    
    def get_weights(self):
        return {
            "gamma": self.gamma.data,
            "beta": self.beta.data,
            "running_mean": self.running_mean.data,
            "running_var": self.running_var.data,
        }

    def set_weights(self, weights):
        self.gamma.data = weights["gamma"]
        self.beta.data = weights["beta"]
        self.running_mean.data = weights.get("running_mean", self.running_mean)
        self.running_var.data = weights.get("running_var", self.running_var)

    

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 weight_init: Callable | str = 'xavier',
                 bias_init: Callable | str = 'zeros',
                 trainable: bool = True,
                 name: str=''):
        super().__init__(name=name, trainable=trainable)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight shape: (KH, KW, C_in, C_out) — same as TensorFlow
        self.weights = self.add_parameter(
            Tensor(resolve_initializer(weight_init, (kernel_size[0], kernel_size[1], in_channels, out_channels)),
                   name='weights'),
            trainable
        )
        self.biases = self.add_parameter(
            Tensor(resolve_initializer(bias_init, (out_channels,)), name='biases'),
            trainable
        )

    def forward(self, x: Tensor) -> Tensor:
        N, H, W, C = x.shape  # NHWC
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1

        # im2col: expect NHWC input → output shape (N, L, KH*KW*C)
        patches = im2col(x, self.kernel_size, self.stride, self.padding)  # (N, L, D)

        # Reshape weights to (D, C_out)
        W = self.weights.reshape((-1, self.out_channels))  # (KH*KW*C_in, C_out)

        # Matrix multiplication
        output = patches @ W  # (N, L, C_out)
        output = output + self.biases  # broadcast (C_out,)

        output = output.reshape((N, H_out, W_out, self.out_channels))  # NHWC
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.weights.shape[2],
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "trainable": self.training
        })
        return config
    
    def get_weights(self):
        return {
            "weights": self.weights.data,
            "biases": self.biases.data,
        }

    def set_weights(self, weights):
        self.weights.data = weights["weights"]
        self.biases.data = weights["biases"]
    

__all__ = list(LAYER_REGISTRY.keys())
