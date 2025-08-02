from .losses import LOSS_REGISTRY, Loss
from .utility import bincount
from .tensor import Tensor

from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any
import numpy as np
import mlx.core as mx


METRIC_REGISTRY = {}

class MetricMeta(ABCMeta):  # <-- Subclass ABCMeta instead of type
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        # Avoid registering the abstract base class itself
        if not cls.__abstractmethods__:
            METRIC_REGISTRY[cls.__name__] = cls

def _tp_fp_fn_tn(pred: Tensor, target: Tensor, class_id: int):
    """Compute TP, FP, FN, TN for a single class."""
    TP = ((pred == class_id) & (target == class_id)).sum().item()
    FP = ((pred == class_id) & (target != class_id)).sum().item()
    FN = ((pred != class_id) & (target == class_id)).sum().item()
    TN = ((pred != class_id) & (target != class_id)).sum().item()
    return TP, FP, FN, TN


class Metric(ABC, metaclass=MetricMeta):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __call__(self, prediction: Tensor, target: Tensor) -> Any:
        x, y = prediction.data, target.data
        if x.size != x.shape[0]:
            if x.ndim != 1:
                x = mx.array.astype(x.argmax(axis=-1, keepdims=False), dtype=mx.int32)
        if y.size != y.shape[0]:
            if y.ndim != 1:
                y = mx.array.astype(y.argmax(axis=-1, keepdims=False), dtype=mx.int32)
        self.forward(x, y)

    @abstractmethod
    def forward(self, prediction: mx.array, target: mx.array):
        pass

    @abstractmethod
    def result(self, prefix: str='') -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
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
    

class MetricManager():
    def __init__(self):
        self.metrics: list[Metric] = []

    def add(self, metric):
        self.metrics.append(metric)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, prediction: Tensor, target: Tensor):
        for metric in self.metrics:
            metric(prediction, target)

    def collect(self, prefix: str='') -> dict[str, float]:
        results = {}
        for metric in self.metrics:
            results.update(metric.result(prefix))  # Each metric returns dict
        return results
    
    def get_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "metrics": [
                {
                    "name": metric.name,
                    "class_name": metric.__class__.__name__,
                    "config": metric.get_config()
                }
                for metric in self.metrics
            ]
        }

    @classmethod
    def from_config(cls, config: dict):
        manager = cls()
        for metric_info in config["metrics"]:
            metric_class = METRIC_REGISTRY[metric_info["class_name"]]
            metric = metric_class.from_config(metric_info["config"])
            manager.add(metric)
        return manager

    
# ============================================================
# Loss
# ============================================================
class LossMetric(Metric):
    def __init__(self, loss_fn: Loss, name='Loss') -> None:
        super().__init__(name)
        self.loss_fn = loss_fn
        self.reset()

    def __call__(self, prediction: Tensor, target: Tensor) -> Any:
        return self.forward(prediction, target)

    def forward(self, prediction: Tensor, target: Tensor):
        loss = self.loss_fn(prediction, target).item()
        self.values.append(loss)

    def result(self, prefix='') -> dict[str, float]:
        value =  sum(self.values) / len(self.values) if self.values else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.values: list[float] = []

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "loss_fn": self.loss_fn.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config) -> Metric:
        config.pop("class_name")
        config["loss_fn"] = LOSS_REGISTRY[config["loss_fn"]["class_name"]].from_config(config["loss_fn"])
        return cls(**config)


# ============================================================
# Accuracy
# ============================================================
class Accuracy(Metric):
    def __init__(self, name: str = "accuracy", average: str = "micro"):
        super().__init__(name)
        assert average in ("micro", "macro", "global")
        self.average = average
        self.reset()

    def forward(self, prediction: mx.array, target: mx.array):
        pred_classes = prediction
        num_classes = int(mx.max(target).item()) + 1

        if self.average in ("micro", "global"):
            self.correct += mx.sum(pred_classes == target).item()
            self.total += target.size
        elif self.average == "macro":
            for k in range(num_classes):
                class_mask = (target == k)
                class_count = mx.sum(class_mask).item()
                if class_count > 0:
                    acc_k = mx.sum((pred_classes == target) & class_mask).item() / class_count
                    self.per_class_acc.append(acc_k)

    def result(self, prefix: str = '') -> dict[str, float]:
        if self.average in ("micro", "global"):
            value = self.correct / self.total if self.total > 0 else 0.0
        else:
            value = sum(self.per_class_acc) / len(self.per_class_acc) if self.per_class_acc else 0.0
        return {prefix + self.name: value}

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_acc = []

    def get_config(self):
        config = super().get_config()
        config.update({"average": self.average})
        return config


class AccuracyForClass(Metric):
    def __init__(self, name: str = "accuracy_class", class_id: int = 0):
        name = f'{name}_{class_id}'
        super().__init__(name)
        self.class_id = class_id
        self.reset()

    def forward(self, prediction: mx.array, target: mx.array):
        class_mask = (target == self.class_id)
        count = mx.sum(class_mask).item()
        if count > 0:
            self.correct += mx.sum((prediction == target) & class_mask).item()
            self.total += count

    def result(self, prefix: str = '') -> dict[str, float]:
        value = self.correct / self.total if self.total > 0 else 0.0
        return {prefix + self.name: value}

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config
    

class AccuracyPerClass(Metric):
    def __init__(self, name: str = "accuracy_class"):
        super().__init__(name)
        self.correct = defaultdict(float)
        self.total = defaultdict(float)

    def forward(self, prediction: mx.array, target: mx.array):
        num_classes = int(mx.max(target).item()) + 1
        correct_mask = prediction == target

        selected = mx.where(correct_mask, target, -1).astype(mx.int32)
        correct_per_class = bincount(selected, num_classes)
        total_per_class = bincount(target.astype(mx.int32), num_classes)

        for k in range(num_classes):
            self.correct[k] += correct_per_class[k].item()
            self.total[k] += total_per_class[k].item()

    def result(self, prefix: str = '') -> dict[str, float]:
        return {
            f"{prefix}{self.name}_{k}": self.correct[k] / self.total[k]
            if self.total[k] > 0 else 0.0
            for k in self.total
        }

    def reset(self):
        self.correct.clear()
        self.total.clear()

    
# ============================================================
# Precision
# ============================================================
class Precision(Metric):
    def __init__(self, name: str = "precision", average: str = "micro"):
        super().__init__(name)
        assert average in ("micro", "macro", "global")
        self.average = average
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        pred_classes = prediction
        num_classes = int(target.max(axis=None).item()) + 1

        if self.average in ("micro", "global"):
            for k in range(num_classes):
                TP, FP, _, _ = _tp_fp_fn_tn(pred_classes, target, k)
                self.tp += TP
                self.fp += FP
        elif self.average == "macro":
            for k in range(num_classes):
                TP, FP, _, _ = _tp_fp_fn_tn(pred_classes, target, k)
                denom = TP + FP
                if denom > 0:
                    self.per_class_prec.append(TP / denom)

    def result(self, prefix: str='') -> dict[str, float]:
        if self.average in ("micro", "global"):
            denom = self.tp + self.fp
            value = self.tp / denom if denom > 0 else 0.0
        # average == 'macro'
        else:
            value =  sum(self.per_class_prec) / len(self.per_class_prec) if self.per_class_prec else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.per_class_prec = []

    def get_config(self):
        config = super().get_config()
        config.update({"average": self.average})
        return config


class PrecisionForClass(Metric):
    def __init__(self, name: str = "precision_class", class_id: int=0):
        name = f'{name}_{class_id}'
        super().__init__(name)
        self.class_id = class_id
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        TP, FP, _, _ = _tp_fp_fn_tn(prediction, target, self.class_id)
        self.tp += TP
        self.fp += FP

    def result(self, prefix: str='') -> dict[str, float]:
        denom = self.tp + self.fp
        value = self.tp / denom if denom > 0 else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fp = 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config
    
class PrecisionPerClass(Metric):
    def __init__(self, name: str = "precision_class"):
        super().__init__(name)
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1
        for k in range(num_classes):
            TP, FP, _, _ = _tp_fp_fn_tn(prediction, target, k)
            self.tp[k] += TP
            self.fp[k] += FP

    def result(self, prefix: str='') -> dict[str, float]:
        return {f"{prefix}{self.name}_{k}": self.tp[k] / (self.tp[k] + self.fp[k]) if (self.tp[k] + self.fp[k]) > 0 else 0.0 for k in self.tp}

    def reset(self):
        self.tp.clear()
        self.fp.clear()

# ============================================================
# Recall
# ============================================================
class Recall(Metric):
    def __init__(self, name: str = "recall", average: str = "micro"):
        super().__init__(name)
        assert average in ("micro", "macro", "global")
        self.average = average
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        pred_classes = prediction
        num_classes = int(target.max(axis=None).item()) + 1

        if self.average in ("micro", "global"):
            for k in range(num_classes):
                TP, _, FN, _ = _tp_fp_fn_tn(pred_classes, target, k)
                self.tp += TP
                self.fn += FN
        elif self.average == "macro":
            for k in range(num_classes):
                TP, _, FN, _ = _tp_fp_fn_tn(pred_classes, target, k)
                denom = TP + FN
                if denom > 0:
                    self.per_class_recall.append(TP / denom)

    def result(self, prefix: str='') -> dict[str, float]:
        if self.average in ("micro", "global"):
            denom = self.tp + self.fn
            value = self.tp / denom if denom > 0 else 0.0
        # average == 'macro'
        else:
            value = sum(self.per_class_recall) / len(self.per_class_recall) if self.per_class_recall else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.per_class_recall = []

    def get_config(self):
        config = super().get_config()
        config.update({"average": self.average})
        return config


class RecallForClass(Metric):
    def __init__(self, name: str = "recall_class", class_id: int=0):
        name = f'{name}_{class_id}'
        super().__init__(name)
        self.class_id = class_id
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        TP, _, FN, _ = _tp_fp_fn_tn(prediction, target, self.class_id)
        self.tp += TP
        self.fn += FN

    def result(self, prefix: str='') -> dict[str, float]:
        denom = self.tp + self.fn
        value = self.tp / denom if denom > 0 else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fn = 0

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config
    

class RecallPerClass(Metric):
    def __init__(self, name: str = "recall_class"):
        super().__init__(name)
        self.tp = defaultdict(int)
        self.fn = defaultdict(int)

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1
        for k in range(num_classes):
            TP, _, FN, _ = _tp_fp_fn_tn(prediction, target, k)
            self.tp[k] += TP
            self.fn[k] += FN

    def result(self, prefix: str='') -> dict[str, float]:
        return {f"{prefix}{self.name}_{k}": self.tp[k] / (self.tp[k] + self.fn[k]) if (self.tp[k] + self.fn[k]) > 0 else 0.0 for k in self.tp}

    def reset(self):
        self.tp.clear()
        self.fn.clear()

# ============================================================
# Specificity
# ============================================================
class Specificity(Metric):
    def __init__(self, name: str = "specificity", average: str = "micro"):
        super().__init__(name=name)
        assert average in ["micro", "macro", "global"], "Invalid averaging type"
        self.average = average
        self.class_stats = {}
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1

        if not self.class_stats:
            self.class_stats = {k: {"tn": 0, "fp": 0} for k in range(num_classes)}

        for k in range(num_classes):
            _, FP, _, TN= _tp_fp_fn_tn(prediction, target, k)
            self.class_stats[k]["tn"] += TN
            self.class_stats[k]["fp"] += FP

    def result(self, prefix: str='') -> dict[str, float]:
        if self.average == "micro":
            tn_total = sum(stats["tn"] for stats in self.class_stats.values())
            fp_total = sum(stats["fp"] for stats in self.class_stats.values())
            denom = tn_total + fp_total
            value = tn_total / denom if denom > 0 else 0.0

        elif self.average == "macro":
            vals = []
            for stats in self.class_stats.values():
                tn, fp = stats["tn"], stats["fp"]
                denom = tn + fp
                vals.append(tn / denom if denom > 0 else 0.0)
            value = sum(vals) / len(vals) if vals else 0.0

        # average == 'global'
        else:
            # Weighted by negatives (tn + fp)
            total_weight = 0
            weighted_sum = 0.0
            for stats in self.class_stats.values():
                tn, fp = stats["tn"], stats["fp"]
                weight = tn + fp
                val = tn / weight if weight > 0 else 0.0
                weighted_sum += val * weight
                total_weight += weight
            value = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {prefix+self.name: value}

    def reset(self):
        self.class_stats = {}

    def get_config(self):
        config = super().get_config()
        config.update({"average": self.average})
        return config
    

class SpecificityForClass(Metric):
    def __init__(self, name: str, class_id: int):
        super().__init__(name=name)
        self.class_id = class_id
        self.tn = 0
        self.fp = 0

    def forward(self, prediction: Tensor, target: Tensor):
        _, FP, _, TN= _tp_fp_fn_tn(prediction, target, self.class_id)
        self.tn += TN
        self.fp += FP

    def result(self, prefix: str='') -> dict[str, float]:
        denom = self.tn + self.fp
        return {prefix+self.name: (self.tn / denom) if denom > 0 else 0.0}

    def reset(self):
        self.tn = 0
        self.fp = 0

    def get_config(self):
        config = super().get_config()
        config.update({"class_id": self.class_id})
        return config

class SpecificityPerClass(Metric):
    def __init__(self, name: str = "specificity_class"):
        super().__init__(name=name)
        self.tn = defaultdict(int)
        self.fp = defaultdict(int)

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1
        for k in range(num_classes):
            _, FP, _, TN= _tp_fp_fn_tn(prediction, target, k)
            self.tn[k] += TN
            self.fp[k] += FP

    def result(self, prefix: str='') -> dict[str, float]:
        return {f"{prefix}{self.name}_{k}": self.tn[k] / (self.tn[k] + self.fp[k]) if (self.tn[k] + self.fp[k]) > 0 else 0.0 for k in self.tn}

    def reset(self):
        self.tn.clear()
        self.fp.clear()

# ============================================================
# F1-Score
# ============================================================
class F1Score(Metric):
    def __init__(self, name: str = "f1_score", average: str = "micro"):
        super().__init__(name=name)
        assert average in ["micro", "macro", "global"], "Invalid averaging type"
        self.average = average
        self.class_stats = {}
        self.reset()

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1

        if not self.class_stats:
            self.class_stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in range(num_classes)}

        if self.average == "micro":
            # Micro = sum TP / (sum TP + sum FP + sum FN)
            TP, FP, FN = 0, 0, 0
            for k in range(num_classes):
                tpk, fpk, fnk, _= _tp_fp_fn_tn(prediction, target, k)
                TP += tpk
                FP += fpk
                FN += fnk
            self.tp += TP
            self.fp += FP
            self.fn += FN

        elif self.average in ["macro", "global"]:
            for k in range(num_classes):
                TP, FP, FN, _= _tp_fp_fn_tn(prediction, target, k)
                self.class_stats[k]["tp"] += TP
                self.class_stats[k]["fp"] += FP
                self.class_stats[k]["fn"] += FN

    def result(self, prefix: str='') -> dict[str, float]:
        if self.average == "micro":
            denom = 2 * self.tp + self.fp + self.fn
            value = (2 * self.tp / denom) if denom > 0 else 0.0

        elif self.average == "macro":
            f1_scores = []
            for stats in self.class_stats.values():
                tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                denom = 2 * tp + fp + fn
                f1_scores.append((2 * tp / denom) if denom > 0 else 0.0)
            value = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # average == 'global'
        else:
            # Weighted by class support (true positives + false negatives)
            total_support = 0
            weighted_sum = 0.0
            for stats in self.class_stats.values():
                tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                support = tp + fn
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                weighted_sum += f1 * support
                total_support += support
            value = weighted_sum / total_support if total_support > 0 else 0.0
        
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.class_stats = {}

    def get_config(self):
        config = super().get_config()
        config.update({"average": self.average})
        return config


class F1ForClass(Metric):
    def __init__(self, name: str = "f1_class", class_id: int=0):
        name = f'{name}_{class_id}'
        super().__init__(name)
        self.class_id = class_id
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def forward(self, prediction: Tensor, target: Tensor):
        TP, FP, FN, _= _tp_fp_fn_tn(prediction, target, self.class_id)
        self.tp += TP
        self.fp += FP
        self.fn += FN

    def result(self, prefix: str='') -> dict[str, float]:
        denom = 2 * self.tp + self.fp + self.fn
        value = (2 * self.tp / denom) if denom > 0 else 0.0
        return {prefix+self.name: value}

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def get_config(self):
        config = super().get_config()
        config.update({"class_id": self.class_id})
        return config


class F1PerClass(Metric):
    def __init__(self, name: str = "f1_class"):
        super().__init__(name)
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def forward(self, prediction: Tensor, target: Tensor):
        num_classes = int(target.max(axis=None).item()) + 1
        for k in range(num_classes):
            TP, FP, FN, _= _tp_fp_fn_tn(prediction, target, k)
            self.tp[k] += TP
            self.fp[k] += FP
            self.fn[k] += FN

    def result(self, prefix: str='') -> dict[str, float]:
        return {f"{prefix}{self.name}_{k}": (2 * self.tp[k] / (2 * self.tp[k] + self.fp[k] + self.fn[k])) if (2 * self.tp[k] + self.fp[k] + self.fn[k]) > 0 else 0.0 for k in self.tp}

    def reset(self):
        self.tp.clear()
        self.fp.clear()
        self.fn.clear()


__all__ = list(METRIC_REGISTRY.keys())
