from .tensor import Tensor
import mlx.core as mx
from typing import Any
from abc import ABC, ABCMeta, abstractmethod

OPTIMISER_REGISTRY = {}

class OptimiserMeta(ABCMeta):  # <-- Subclass ABCMeta instead of type
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Avoid registering the abstract base class itself
        if not cls.__abstractmethods__:
            OPTIMISER_REGISTRY[cls.__name__] = cls

class Optimiser(ABC, metaclass=OptimiserMeta):
    def __init__(self, learning_rate: float = 0.01, name: str = ''):
        assert learning_rate > 0, "Learning rate must be positive"
        self.learning_rate = learning_rate
        self.name = name or self.__class__.__name__.lower()

    @abstractmethod
    def apply_gradients(self, params: list[Tensor]):
        """Update the parameters based on their gradients."""
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.__class__.__name__,
            "learning_rate": self.learning_rate
        }

    @classmethod
    def from_config(cls, config: dict):
        config.pop("class_name")
        return cls(**config)


class SGD(Optimiser):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False, name: str = ''):
        super().__init__(learning_rate, name)
        assert momentum >= 0, "Momentum must be non-negative"
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.velocities: dict[int, mx.array] = {}  # param_id -> velocity Tensor

    def apply_gradients(self, params: list[Tensor]):
        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            param_id = id(param)

            if self.momentum > 0:
                if param_id not in self.velocities:
                    self.velocities[param_id] = mx.zeros_like(grad)

                v = self.velocities[param_id]
                v_new = self.momentum * v + grad  # store raw gradient influence
                self.velocities[param_id] = v_new

                if self.nesterov:
                    # Look ahead
                    step = grad + self.momentum * v_new
                else:
                    step = v_new

                # Apply step scaled by learning rate
                param.update(self.learning_rate * step)
            else:
                # Standard SGD
                param.update(self.learning_rate * grad)


    def get_config(self):
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "nesterov": self.nesterov
        })
        return config
    

class Adam(Optimiser):
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 name: str = ''):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State: param_id -> (m, v)
        self.m: dict[int, mx.array] = {}
        self.v: dict[int, mx.array] = {}
        self.t: int = 0

    def apply_gradients(self, params: list[Tensor]):
        self.t += 1  # increment timestep

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            param_id = id(param)

            # Initialize state if first time
            if param_id not in self.m:
                self.m[param_id] = mx.zeros_like(grad)
                self.v[param_id] = mx.zeros_like(grad)

            m = self.m[param_id]
            v = self.v[param_id]

            # Update biased first and second moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            # Store updated moments
            self.m[param_id] = m
            self.v[param_id] = v

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Compute update
            update = self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.epsilon)

            # Apply update (gradient descent → subtract update)
            param.update(update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        })
        return config


__all__ = list(OPTIMISER_REGISTRY.keys())
