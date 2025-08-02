from .tensor import Tensor
import mlx.core as mx
from abc import ABC, ABCMeta, abstractmethod
from typing import Any

LOSS_REGISTRY = {}

class LossMeta(ABCMeta):  # <-- Subclass ABCMeta instead of type
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Avoid registering the abstract base class itself
        if not cls.__abstractmethods__:
            LOSS_REGISTRY[cls.__name__] = cls

def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    data = labels.data.astype(mx.int32)  # or mx.int64 depending on your platform/setup
    out = mx.zeros((data.size, num_classes))
    out[mx.arange(data.size), data] = 1.0
    return Tensor(out, requires_grad=False)


class Loss(ABC, metaclass=LossMeta):
    def __init__(self, name: str='') -> None:
        super().__init__()
        self.name = name if name else self.__class__.__name__

    def __call__(self, prediction: Tensor, target: Tensor):
        return self.forward(prediction, target)

    @abstractmethod
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pass
    
    def get_config(self) -> dict[str, Any]:
        config = {
            "name": self.name,
            "class_name": self.__class__.__name__,
        }
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        config.pop("class_name")
        return cls(**config)


class MSELoss(Loss):
    def __init__(self, name: str = '') -> None:
        super().__init__(name)

    def forward(self, prediction: Tensor, target: Tensor):
        diff = prediction - target
        return (diff * diff).mean(axis=None)


class CrossEntropyLoss(Loss):
    def __init__(self, label_smoothing: float = 0.0, name: str = ''):
        super().__init__(name)
        self.label_smoothing = label_smoothing

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction._name != 'softmax':
            UserWarning('CrossEntropyLoss used without softmax output. Is this intentional?')

        batch_size, num_classes = prediction.shape

        # If target is 1D (class indices), convert to one-hot
        if len(target.shape) == 1:
            one_hot_targets = one_hot(target, num_classes).data
        else:
            one_hot_targets = target.data  # Assume already one-hot or smoothed

        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            smooth = self.label_smoothing
            one_hot_targets = (1 - smooth) * one_hot_targets + smooth / num_classes

        target_tensor = Tensor(one_hot_targets, requires_grad=False)

        log_preds = prediction.log(base=mx.e)
        loss_per_sample = - (target_tensor * log_preds).sum(axis=1)
        return loss_per_sample.mean(axis=None)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({'label_smoothing': self.label_smoothing})
        return config



__all__ = list(LOSS_REGISTRY.keys())
