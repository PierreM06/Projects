import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from tensorgraph import Tensor
import mlx.core as mx
import csv
import time

from tensorgraph.models import Sequential
from tensorgraph.layers import Dense, Activation
from tensorgraph.losses import CrossEntropyLoss
from tensorgraph.optimisers import SGD
from tensorgraph.metrics import Accuracy, AccuracyPerClass


# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten to shape (num_samples, 784)
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Convert numpy arrays to mx arrays, then wrap in your Tensor class
X_train = Tensor(mx.array(x_train))
y_train = Tensor(mx.array(y_train))

X_test = Tensor(mx.array(x_test))
y_test = Tensor(mx.array(y_test))

loss_fn = CrossEntropyLoss()
optimiser = SGD()
model = Sequential(loss_fn, optimiser)
model.add([
    Dense(784, 128),
    Activation('relu'),
    Dense(128, 64),
    Activation('relu'),
    Dense(64, 10),
    Activation('softmax'),

    Accuracy(),
    AccuracyPerClass()
])

times = defaultdict(list)
for i in range(8,16):
    for _ in range(10):
        start = time.time()
        # Train the model
        metrics = model.fit(
            X_train, y_train,
            epoch=10,
            batch_size=2**i,
            validation=(X_test, y_test)
        )
        times[2**i].append((time.time() - start)/10)

times = {k: sum(v)/len(v) for k, v in times.items()}
print(times)
quit()

keys = list(metrics.keys())
rows = zip(*[metrics[key] for key in keys])  # transpose

with open('metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(keys)  # header
    writer.writerows(rows)
