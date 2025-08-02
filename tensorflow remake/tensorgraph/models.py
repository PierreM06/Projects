from .metrics import METRIC_REGISTRY, MetricManager, Metric, LossMetric
from .optimisers import OPTIMISER_REGISTRY, Optimiser
from .utility import EarlyStopping, ProgressBar
from .layers import LAYER_REGISTRY, Layer
from .losses import LOSS_REGISTRY, Loss
from .visualiser import graph
from .tensor import Tensor

from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Iterable
from math import ceil
import mlx.core as mx
import numpy as np
import json

METRIC_REGISTRY["MetricManager"] = MetricManager
MODEL_REGISTRY = {}

class ModelMeta(ABCMeta):  # <-- Subclass ABCMeta instead of type
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Avoid registering the abstract base class itself
        if not cls.__abstractmethods__:
            MODEL_REGISTRY[cls.__name__] = cls

"""
TODO: 
- Implement __str__ methods for better model representation.
- Implement more complex models with different layer types.
- Add support for saving and loading models.
- Add support for custom loss, activation, layers, model, optimizers, etc functions.
- Implement early stopping based on validation loss.
- Add support for auto input shape inference in layers.
- Implement a method to visualize the model architecture.
- Expand on visualiser to switch direction of arrows, and more.
- Add more models, losses, optimizers, and layers.
- Make base Layer, Model, Metric, Optimiser ans loss ABC classes.
- Make save/load test file, check size of saved files in tests.
- Fix CrossEntropy loss returning shape [1,1]
"""

class Model(ABC, metaclass=ModelMeta):
    def __init__(self, 
                 loss: Loss, 
                 optimiser: Optimiser, 
                 early_stopping: EarlyStopping | None = None, 
                 seed: int=None) -> None:
        super().__init__()
        if seed is not None:
            mx.random.seed(seed)

        self.loss = loss
        self.optimiser = optimiser
        self.early_stopping = early_stopping
        self._early_stopping: bool = False
        self.layers: list[Layer] = []
        self.metric_manager: MetricManager = MetricManager()
        self.metric_manager.add(LossMetric(loss, loss.name))
        if self.early_stopping != None and self.early_stopping.monitor == '':
            self.early_stopping.monitor = loss.name

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def add(self, objects: Layer | Metric | Iterable[Layer | Metric]):
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backwards(self, input: Tensor, target: Tensor):
        pass

    def on_epoch_start(self, epoch: int): pass
    def on_epoch_end(self, epoch: int, progress_bar: ProgressBar, logs: defaultdict, epoch_logs: dict) -> defaultdict: pass

    def on_batch_start(self, batch: int): pass
    def on_batch_end(self, batch: int) -> dict: pass

    def on_validation_start(self): pass
    def on_validation_end(self) -> dict: pass
    
    @abstractmethod
    def fit(self, 
            inputs: Tensor, 
            targets: Tensor, 
            epoch: int, 
            batch_size: int = 1, 
            validation: tuple[Tensor, Tensor] | None = None,
            viz: bool = False
            ) -> dict[str, list[float]]:
        pass
    
    def train(self):
        for layer in self.layers:
            layer.set_training(True)

    def eval(self):
        for layer in self.layers:
            layer.set_training(False)

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def update(self):
        self.optimiser.apply_gradients(self.parameters())
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "layers": [layer.get_config() for layer in self.layers],
            "loss": self.loss.get_config(),
            "optimiser": self.optimiser.get_config(),
            "metrics": self.metric_manager.get_config()
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "Model":
        # You can use a MODEL_REGISTRY here if needed
        model = cls.__new__(cls)  # allocate instance without calling __init__
        super(cls, model).__init__(
            loss=LOSS_REGISTRY[config["loss"]["class_name"]].from_config(config["loss"]),
            optimiser=OPTIMISER_REGISTRY[config["optimiser"]["class_name"]].from_config(config["optimiser"]),
        )

        # Set up layers
        model.layers = [
            LAYER_REGISTRY[layer_cfg["class_name"]].from_config(layer_cfg)
            for layer_cfg in config["layers"]
        ]

        # Set up metric manager
        model.metric_manager = MetricManager.from_config(config["metrics"])

        return model
    
    def save_weights(self, path: str):
        weights_to_save = {}
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            for name, tensor in layer_weights.items():
                weights_to_save[f"layer{i}_{name}"] = tensor
        mx.savez(path, **weights_to_save)
        
    def save_config(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2)
    

def save_model(model: Model, path: str):
    """
    _summary_

    Args:
        model (Model): _description_
        path (str): Path to save the config and weights, do NOT put a file extension it gets added.
    """
    config_path = f"{path}.json"
    weights_path = f"{path}.npz"

    model.save_config(config_path)
    model.save_weights(weights_path)

def _load_config(path: str) -> Model:
    with open(path, "r") as f:
        config = json.load(f)
    return MODEL_REGISTRY[config['class_name']].from_config(config)  # or resolve from class_name if you support multiple model types

def _load_weights(model, path):
    loaded = mx.load(path)
    for i, layer in enumerate(model.layers):
        layer_weights = {}
        # Match keys for this layer
        for key in loaded:
            prefix = f"layer{i}_"
            if key.startswith(prefix):
                name = key[len(prefix):]
                layer_weights[name] = loaded[key]
        layer.set_weights(layer_weights)

def load_model(path: str) -> Model:
    config_path = f"{path}.json"
    weights_path = f"{path}.npz"

    model = _load_config(config_path)
    _load_weights(model, weights_path)
    return model


class Sequential(Model):
    def __init__(self, 
                 loss: Loss, 
                 optimiser: Optimiser, 
                 early_stopping: EarlyStopping | None=None, 
                 seed: int=None) -> None:
        super().__init__(loss, optimiser, early_stopping, seed)


    def add(self, objects: Layer | Metric | Iterable[Layer | Metric]):
        if not isinstance(objects, Iterable) or isinstance(objects, (Layer, Metric)):
            objects = [objects]  # Wrap single object in a list

        for obj in objects:
            if isinstance(obj, Layer):
                self.layers.append(obj)
            elif isinstance(obj, Metric):
                self.metric_manager.add(obj)
            else:
                raise TypeError(f"Cannot add object of type {type(obj)} to the model.")


    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        return self.forward(x)
    
    def backwards(self, input: Tensor, target: Tensor, viz: bool=False):
        output = self.forward(input)
        if output[0][0].item() == mx.nan:
            pass

        loss = self.loss(output, target)
        loss.backward()
        if viz:
            graph(loss)

        self.update()

        self.metric_manager.update(output, target)

    def on_epoch_start(self, epoch: int): 
        self.train()
        self.metric_manager.reset()
    
    def on_epoch_end(self, epoch: int, progress_bar: ProgressBar, logs: defaultdict, epoch_logs: dict) -> defaultdict: 
        for key, val in epoch_logs.items():
            logs[key].append(val)

        if self.early_stopping is not None:
            monitor_value = epoch_logs.get(self.early_stopping.monitor, None)
            if monitor_value is not None and self.early_stopping.check(monitor_value, epoch):
                print(f"Early stopping at epoch {epoch + 1} — best {self.early_stopping.monitor}: {self.early_stopping.best}")
                self._early_stop = True  # Set flag to break main loop

        progress_bar()
        return logs

    def on_validation_start(self):
        self.eval()
        self.metric_manager.reset()

    def on_validation_end(self) -> dict: pass

    def fit(self, 
            inputs: Tensor, 
            targets: Tensor, 
            epoch: int, 
            batch_size: int = 1, 
            validation: tuple[Tensor, Tensor] | None = None,
            viz: bool = False
            ) -> dict[str, list[float]]:
        
        if batch_size > len(inputs) or batch_size == -1:
            batch_size = len(inputs)

        amount_of_batches = ceil(len(inputs) / batch_size)
        if validation:
            val_inputs, val_targets = validation
            amount_of_val_batches = ceil(len(val_inputs) / batch_size)

        metrics: dict[str, list[float]] = defaultdict(list)
        self.metric_manager.reset()

        progress = ProgressBar(max=epoch, title=f'Epoch')
        for e in range(epoch):
            self.on_epoch_start(e)

            for j in range(amount_of_batches):
                batch_indecies = (batch_size*j), min(batch_size*(j+1), len(inputs))
                input_batch = inputs[batch_indecies[0]:batch_indecies[1]]
                target_batch = targets[batch_indecies[0]:batch_indecies[1]]

                self.backwards(input_batch, target_batch, viz=viz)
                self.zero_grad()

            epoch_metrics = self.metric_manager.collect()

            if not validation:
                metrics = self.on_epoch_end(e, progress, metrics, epoch_metrics)
                continue

            self.on_validation_start()
        
            for j in range(amount_of_val_batches):
                val_batch_indecies = (batch_size*j), min(batch_size*(j+1), len(val_inputs))
                val_input_batch = val_inputs[val_batch_indecies[0]:val_batch_indecies[1]]
                val_target_batch = val_targets[val_batch_indecies[0]:val_batch_indecies[1]]

                val_preds = self.forward(val_input_batch)

                self.loss(val_preds, val_target_batch)
                self.metric_manager.update(val_preds, val_target_batch)

            epoch_metrics.update(self.metric_manager.collect(prefix='val_'))

            metrics = self.on_epoch_end(e, progress, metrics, epoch_metrics)

            if self._early_stopping:
                break

        return dict(metrics)
    
    def __str__(self) -> str:
        description = f"Sequential Model with {len(self.layers)} layers:\n"
        for i, layer in enumerate(self.layers):
            description += f"  [{i}] {layer}\n"
        return description
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
                        "layers": [layer.get_config() for layer in self.layers],
        })
        return config


__all__ = list(MODEL_REGISTRY.keys())
