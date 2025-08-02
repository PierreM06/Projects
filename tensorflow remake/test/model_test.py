from tensorgraph.models import save_model, load_model, Sequential
from tensorgraph.visualiser import visualize_tensor_graph
from tensorgraph.layers import Dense, Activation
from tensorgraph.utility import ProgressBar
from tensorgraph.optimisers import SGD
from tensorgraph.tensor import Tensor
import tensorflow as tf
from tensorgraph.losses import MSELoss
import mlx.core as mx
import pandas as pd
import numpy as np
import tempfile
import pytest
import os

def human_readable_size(bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024

def test_save_and_load_model():
    # Setup temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        # Step 1: Create model
        model = Sequential(loss=MSELoss(), optimiser=SGD(learning_rate=0.01))
        model.add(Dense(4, 8))
        model.add(Activation("relu"))
        model.add(Dense(8, 1))
        
        # Step 2: Dummy data
        x = Tensor(mx.random.uniform(-1, 1, (10, 4)))
        y = Tensor(mx.random.uniform(-1, 1, (10, 1)))

        model.fit(x, y, epoch=2, batch_size=2)

        # Step 3: Save model
        save_model(model, model_path)

        # Step 4: Load model
        loaded_model = load_model(model_path)

        # Step 5a: Compare config
        assert model.get_config() == loaded_model.get_config(), "Configs do not match"

        # Step 5b: Compare output
        original_output = model.predict(x)
        loaded_output = loaded_model.predict(x)

        assert  mx.allclose(original_output.data, loaded_output.data, atol=1e-7), f"Model output mismatch after load"

        print(human_readable_size(os.path.getsize(model_path+".json")))
        print(human_readable_size(os.path.getsize(model_path+".npz")))
        print("✅ Save/load test passed successfully.")

def test_and_nn(debug=False):
    W1 = [[-0.5], [0.5]]  # weights for AND gate
    b1 = [1.5]  # bias for AND gate

    and_tf = tf.keras.Sequential([
        tf.keras.Input(shape=[2], dtype=tf.float32),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, 
                              kernel_initializer=tf.keras.initializers.Constant(W1), 
                              bias_initializer=tf.keras.initializers.Constant(b1))]
    )
    and_tf.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), 
        loss='mean_squared_error'
    )

    and_nn = Sequential(MSELoss(), SGD(1))
    and_nn.add(Dense(1,2, weight_init=W1, bias_init=b1))
    and_nn.add(Activation('sigmoid'))

    input_data = Tensor(mx.array([[0,0],[1,0],[0,1],[1,1]]), name='input_data')
    target = Tensor(mx.array([[0],[0],[0],[1]]), name='target_data')  # Expected outputs for AND gate

    results = []
    progress = ProgressBar(max=100, title="Training AND NN")
    for epoch in range(100):  # Simulate training
        for i in range(len(input_data)):
            inp, t = input_data[i].reshape((1,-1)), target[i]

            out_tf = and_tf.predict(inp.numpy(), verbose=0).tolist()
            loss_tf = np.mean((out_tf - t.numpy()) ** 2)
            and_tf.train_on_batch(inp.numpy(), t.numpy())

            and_nn.backwards(inp, t)

            and_nn.zero_grad()

            if debug:
                results.append({
                    'iteration': epoch*4 + i,
                    'output_tf': out_tf[-1][-1],
                    'loss_tf': loss_tf,
                    'output_nn': and_nn.layers[-1].last_output.tolist()[0][0],
                })

            assert mx.allclose(mx.array(out_tf[-1][-1]), and_nn.layers[-1].last_output.data, atol=1e-7), f"Output mismatch between tensorflow and MLX model, {out_tf} and {and_nn.layers[-1].last_output.data}"

        progress()

    and_nn.fit(input_data, target, 1000, viz=debug)

    if debug:
        df = pd.DataFrame(results)
        df.to_csv('and_results.csv', index=False)

        # visualize_tensor_graph(and_nn.last_loss)


def test_xor_nn(debug=False):
    W1 = [[0.2, 0.7], [-0.4, 0.1]]  # weights for XOR gate
    b1 = [0.0, 0.0]  # biases for XOR

    W2 = [[0.6], [0.9]]  # weights for XOR gate
    b2 = [0.0]  # bias for XOR gate

    xor_tf = tf.keras.Sequential([
        tf.keras.Input(shape=[2], dtype=tf.float32),
        tf.keras.layers.Dense(2, activation='sigmoid', use_bias=True, 
                              kernel_initializer=tf.keras.initializers.Constant(W1), 
                              bias_initializer=tf.keras.initializers.Constant(b1)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, 
                              kernel_initializer=tf.keras.initializers.Constant(W2), 
                              bias_initializer=tf.keras.initializers.Constant(b2))]
    )
    xor_tf.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), 
        loss='mean_squared_error'
    )

    xor_nn = Sequential(MSELoss(), SGD(1))
    xor_nn.add(Dense(2, 2, weight_init=W1, bias_init=b1))
    xor_nn.add(Activation('sigmoid'))
    xor_nn.add(Dense(1, 2, weight_init=W2, bias_init=b2))
    xor_nn.add(Activation('sigmoid'))

    input_data = Tensor(mx.array([[0,0],[1,0],[0,1],[1,1]]), name='input_data')
    target = Tensor(mx.array([[0],[1],[1],[0]]), name='target_data')

    results = []
    progress = ProgressBar(max=100, title="Training XOR NN")
    for epoch in range(100):  # Simulate training
        for i in range(len(input_data)):
            inp, t = input_data[i].reshape((1,-1)), target[i]

            out_tf = xor_tf.predict(inp.numpy(), verbose=0).tolist()
            xor_tf.train_on_batch(inp.numpy(), t.numpy())
            loss_tf = np.mean((out_tf - t.numpy()) ** 2)

            xor_nn.backwards(inp, t)

            xor_nn.zero_grad()

            if debug:
                results.append({
                    'iteration': epoch*4 + i,
                    'output_tf': out_tf[-1][-1],
                    'loss_tf': loss_tf,
                    'output_nn': xor_nn.layers[-1].last_output.tolist()[0][0],
                })

            assert mx.allclose(mx.array(out_tf[-1][-1]), xor_nn.layers[-1].last_output.data, atol=1e-7), f"Output mismatch between tensorflow and MLX model, {out_tf} and {xor_nn.layers[-1].last_output.data}"

        progress()

    xor_nn.fit(input_data, target, 1000)

    if debug:
        df = pd.DataFrame(results)
        df.to_csv('xor_results.csv', index=False)

        # visualize_tensor_graph(xor_nn.last_loss)


def test_adder_nn(debug=False):
    # Hidden layer weights (3 neurons, 2 inputs)
    W1 = [
        [0.5, -0.4,  0.3],   # weights from input neuron 0 to hidden neurons 0, 1, 2
        [-0.6, 0.9, -0.7]    # weights from input neuron 1 to hidden neurons 0, 1, 2
    ]  # shape: (2, 3)
    b1 = [0.1, -0.2, 0.3]    # one bias per hidden neuron

    W2 = [
        [ 0.8,  0.4],  # weights from hidden neuron 0 to output 0 (sum), output 1 (carry)
        [-0.5,  0.7],  # hidden neuron 1
        [-0.6,  0.2]   # hidden neuron 2
    ]  # shape: (3, 2)
    b2 = [0.0, 0.1]  # one bias per output neuron

    half_adder_tf = tf.keras.Sequential([
        tf.keras.Input(shape=[2], dtype=tf.float32),
        tf.keras.layers.Dense(3, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.Constant(W1),
            bias_initializer=tf.keras.initializers.Constant(b1)),
        tf.keras.layers.Dense(2, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.Constant(W2),
            bias_initializer=tf.keras.initializers.Constant(b2)),
    ])

    half_adder_tf.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss='mean_squared_error'
    )

    half_adder_nn = Sequential(MSELoss(), SGD(1))
    half_adder_nn.add(Dense(3, 2, weight_init=W1, bias_init=b1))
    half_adder_nn.add(Activation('sigmoid'))
    half_adder_nn.add(Dense(2, 3, weight_init=W2, bias_init=b2))
    half_adder_nn.add(Activation('sigmoid'))

    input_data = Tensor(mx.array([[0, 0], [0, 1], [1, 0], [1, 1]]), name='input_data')
    target_data = Tensor(mx.array([[0, 0], [1, 0], [1, 0], [0, 1]]), name='target_data')

    results = []
    progress = ProgressBar(max=100, title="Training Half Adder NN")
    for epoch in range(100):
        # TensorFlow training
        out_tf = half_adder_tf(input_data.numpy())
        half_adder_tf.train_on_batch(input_data.numpy(), target_data.numpy())
        loss_tf = np.mean((out_tf - target_data.numpy()) ** 2)

        # Custom model training
        half_adder_nn.backwards(input_data, target_data)

        half_adder_nn.zero_grad()

        if debug:
            results.append({
                'iteration': epoch,
                'output_tf': np.array(out_tf),
                'loss_tf': loss_tf,
                'output_nn': half_adder_nn.layers[-1].last_output.tolist(),
            })

        # Save or print comparisons here
        assert mx.allclose(mx.array(out_tf), half_adder_nn.layers[-1].last_output.data, atol=1e-5), f"Output mismatch between tensorflow and MLX model, {out_tf} and {half_adder_nn.layers[-1].last_output.data}"

        progress()

    half_adder_nn.fit(input_data, target_data, 1000, 3)

    if debug:
        df = pd.DataFrame(results)
        df.to_csv('adder_results.csv', index=False)

        # visualize_tensor_graph(half_adder_nn.last_loss)

if __name__ == "__main__":
    test_save_and_load_model()
    test_and_nn(debug=True)
    test_xor_nn(debug=True)
    test_adder_nn(debug=True)
