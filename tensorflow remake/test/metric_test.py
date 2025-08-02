import mlx.core as mx
import pytest
from tensorgraph.tensor import Tensor
from tensorgraph.metrics import (
    Accuracy, AccuracyForClass, AccuracyPerClass,
    Precision, PrecisionForClass, PrecisionPerClass,
    Recall, RecallForClass, RecallPerClass,
    F1Score, F1ForClass, F1PerClass,
    Specificity, SpecificityForClass, SpecificityPerClass,
)

# Utility to wrap mlx array into Tensor
def T(arr):
    return Tensor(mx.array(arr))

# Example batch: 6 samples, 3 classes
preds = T([0, 1, 2, 2, 1, 0])
targets = T([0, 1, 2, 1, 1, 0])  # Mistake: pred[3] = 2 instead of 1

# -----------------------------------------------------------
# Accuracy
# -----------------------------------------------------------
def test_accuracy_global():
    metric = Accuracy("accuracy_global", average="global")
    metric(preds, targets)
    # 5 correct out of 6
    assert abs(metric.result()['accuracy_global'] - (5/6)) < 1e-6

def test_accuracy_micro():
    metric = Accuracy("accuracy_micro", average="micro")
    metric(preds, targets)
    # micro accuracy = same as global here (same formula)
    assert abs(metric.result()['accuracy_micro'] - (5/6)) < 1e-6

def test_accuracy_macro():
    metric = Accuracy("accuracy_macro", average="macro")
    metric(preds, targets)
    # Per-class: 0:1.0, 1:2/3, 2:1.0 => avg = (1 + 2/3 + 1)/3
    expected = (1 + 2/3 + 1) / 3
    assert abs(metric.result()['accuracy_macro'] - expected) < 1e-6

# Accuracy for a single class
def test_accuracy_for_class():
    metric = AccuracyForClass(name="accuracy_class", class_id=1)
    metric(preds, targets)
    # Class 1 actual positions: 1,3,4 -> predicted correctly at 1,4
    expected = 2/3
    assert abs(metric.result()['accuracy_class_1'] - expected) < 1e-6

# Accuracy per class (dict)
def test_accuracy_per_class():
    metric = AccuracyPerClass("accuracy_class")
    metric(preds, targets)
    res = metric.result()
    assert abs(res["accuracy_class_0"] - 1.0) < 1e-6
    assert abs(res["accuracy_class_1"] - (2/3)) < 1e-6
    assert abs(res["accuracy_class_2"] - 1.0) < 1e-6

# -----------------------------------------------------------
# Precision
# -----------------------------------------------------------
def test_precision_global():
    metric = Precision("precision_global", average="global")
    metric(preds, targets)
    # TP = 5, predicted = 6
    assert abs(metric.result()["precision_global"] - (5/6)) < 1e-6

def test_precision_micro():
    metric = Precision("precision_micro", average="micro")
    metric(preds, targets)
    # Micro = same as global in this context
    assert abs(metric.result()["precision_micro"] - (5/6)) < 1e-6

def test_precision_macro():
    metric = Precision("precision_macro", average="macro")
    metric(preds, targets)
    # Per-class:
    # class0: TP=2, pred=2 => 1.0
    # class1: TP=2, pred=2 => 1.0
    # class2: TP=1, pred=2 => 0.5
    expected = (1.0 + 1.0 + 0.5) / 3
    assert abs(metric.result()["precision_macro"] - expected) < 1e-6

def test_precision_for_class():
    metric = PrecisionForClass(name="prec_class", class_id=1)
    metric(preds, targets)
    expected = 1.0  # Predicted 1 only at [1,4], both correct
    assert abs(metric.result()["prec_class_1"] - expected) < 1e-6

def test_precision_per_class():
    metric = PrecisionPerClass("precision_class")
    metric(preds, targets)
    res = metric.result()
    assert abs(res["precision_class_0"] - 1.0) < 1e-6
    assert abs(res["precision_class_1"] - 1.0) < 1e-6
    assert abs(res["precision_class_2"] - 0.5) < 1e-6

# -----------------------------------------------------------
# Recall
# -----------------------------------------------------------
def test_recall_global():
    metric = Recall("recall_global", average="global")
    metric(preds, targets)
    # TP=5, actual positives=6
    assert abs(metric.result()["recall_global"] - (5/6)) < 1e-6

def test_recall_micro():
    metric = Recall("recall_micro", average="micro")
    metric(preds, targets)
    assert abs(metric.result()["recall_micro"] - (5/6)) < 1e-6

def test_recall_macro():
    metric = Recall("recall_macro", average="macro")
    metric(preds, targets)
    # Per-class recall:
    # class0: 1.0 (2/2)
    # class1: 2/3
    # class2: 1.0 (1/1)
    expected = (1 + 2/3 + 1) / 3
    assert abs(metric.result()["recall_macro"] - expected) < 1e-6

def test_recall_for_class():
    metric = RecallForClass(name="recall_class", class_id=1)
    metric(preds, targets)
    expected = 2/3
    assert abs(metric.result()["recall_class_1"] - expected) < 1e-6

def test_recall_per_class():
    metric = RecallPerClass("recall_class")
    metric(preds, targets)
    res = metric.result()
    assert abs(res["recall_class_0"] - 1.0) < 1e-6
    assert abs(res["recall_class_1"] - (2/3)) < 1e-6
    assert abs(res["recall_class_2"] - 1.0) < 1e-6

# -----------------------------------------------------------
# Specificity
# -----------------------------------------------------------
def test_specificity_global():
    metric = Specificity("specificity", average="global")
    metric(preds, targets)
    expected = 11/12
    assert abs(metric.result()["specificity"] - expected) < 1e-6

def test_specificity_micro():
    metric = Specificity("specificity", average="micro")
    metric(preds, targets)
    expected = 11/12
    assert abs(metric.result()["specificity"] - expected) < 1e-6

def test_specificity_macro():
    metric = Specificity("specificity", average="macro")
    metric(preds, targets)
    expected = (1 + 1 + 0.8) / 3
    assert abs(metric.result()["specificity"] - expected) < 1e-6

def test_specificity_for_class():
    metric = SpecificityForClass("specificity_class_1", class_id=1)
    metric(preds, targets)
    expected = 1
    assert abs(metric.result()["specificity_class_1"] - expected) < 1e-6

def test_specificity_per_class():
    metric = SpecificityPerClass()
    metric(preds, targets)
    res = metric.result()
    assert abs(res["specificity_class_0"] - 1.0) < 1e-6
    assert abs(res["specificity_class_1"] - 1.0) < 1e-6
    assert abs(res["specificity_class_2"] - (4/5)) < 1e-6

# -----------------------------------------------------------
# F1 Score
# -----------------------------------------------------------
def test_f1_micro():
    metric = F1Score("f1_micro", average="micro")
    metric(preds, targets)
    expected = 10 / 12 
    result = metric.result()
    assert abs(result["f1_micro"] - expected) < 1e-6

def test_f1_macro():
    metric = F1Score("f1_macro", average="macro")
    metric(preds, targets)
    expected = (1.0 + 0.8 + (2/3)) / 3
    result = metric.result()
    assert abs(result["f1_macro"] - expected) < 1e-6

def test_f1_global():
    metric = F1Score("f1_global", average="global")
    metric(preds, targets)
    expected = (2*1.0 + 3*0.8 + 1*(2/3)) / 6
    result = metric.result()
    assert abs(result["f1_global"] - expected) < 1e-6

def test_f1_for_class():
    metric = F1ForClass(class_id=1)
    metric(preds, targets)
    expected = 0.8
    result = metric.result()
    assert abs(result["f1_class_1"] - expected) < 1e-6

def test_f1_per_class():
    metric = F1PerClass()
    metric(preds, targets)
    result = metric.result()
    assert abs(result["f1_class_0"] - 1.0) < 1e-6
    assert abs(result["f1_class_1"] - 0.8) < 1e-6
    assert abs(result["f1_class_2"] - (2/3)) < 1e-6


if __name__ == '__main__':
    test_accuracy_per_class()
