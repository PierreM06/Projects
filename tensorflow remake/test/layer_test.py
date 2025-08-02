import mlx.core as mx
from tensorgraph.tensor import Tensor
from tensorgraph.layers import Layer, Dense, BatchNorm, Activation, Conv2D
from tensorgraph.visualiser import graph, visualize_tensor_graph
import tensorflow as tf
from typing import Callable, Any


def compare_with_tf(
    custom_layer: Layer,
    tf_layer_fn,
    input_tensor: Tensor,
    tf_param_map: dict[str, Callable[[Any], tf.Tensor]],
    atol=1e-4,
    debug: bool=False
):
    out = custom_layer(input_tensor)
    loss = out.sum()
    loss.backward()

    if debug:
        graph(loss)
        visualize_tensor_graph(loss)

    # TF input: NHWC already
    tf_input = tf.Variable(tf.convert_to_tensor(input_tensor.numpy()))

    # Extract and map weights/biases from your layer
    param_data = {p._name: p.numpy() for p in custom_layer.parameters()}

    # Instantiate the TensorFlow layer using tf_layer_fn
    tf_layer = tf_layer_fn(param_data)

    # TF forward + backward
    with tf.GradientTape() as tape:
        out_tf = tf_layer(tf_input)
        loss_tf = tf.reduce_sum(out_tf)

    tf_grads = tape.gradient(loss_tf, [tf_input] + [tf_param_map[name](tf_layer) for name in tf_param_map])

    # Compare forward output
    out_tf_np = out_tf.numpy()
    assert mx.allclose(out.data, mx.array(out_tf_np), atol=atol), f"Forward output mismatch, expected {mx.array(out_tf_np)}, got {out.grad}"

    # Compare input gradient
    grad_input_tf = tf_grads[0].numpy()
    assert mx.allclose(input_tensor.grad, mx.array(grad_input_tf), atol=atol), f"Input gradient mismatch, expected {mx.array(grad_input_tf)}, got {input_tensor.grad}"

    # Compare all parameter gradients
    for i, param in enumerate(custom_layer.parameters()):
        grad_tf = tf_grads[i + 1].numpy()
        assert mx.allclose(param.grad, mx.array(grad_tf), atol=atol), f"Gradient mismatch for parameter '{param._name}', expected {mx.array(grad_tf)}, got {param.grad}"


def test_batchnorm_forward_backward():
    # Input
    x_data = mx.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    x = Tensor(x_data, requires_grad=True, name="input")

    # Layer setup
    bn = BatchNorm(dim=2)
    bn.gamma.data = mx.ones((1, 2))
    bn.beta.data = mx.zeros((1, 2))

    # Forward pass
    out = bn.forward(x)

    # Manual normalization
    mean = mx.mean(x_data, axis=0, keepdims=True)
    var = mx.var(x_data, axis=0, keepdims=True)
    norm = (x_data - mean) / mx.sqrt(var + bn.epsilon)

    assert mx.allclose(out.data, norm, atol=1e-5), f"Forward check failed. Expected {norm}, got {out.data}"

    # Backward pass
    loss = out.sum()
    out.backward()
    # loss.backward()

    visualize_tensor_graph(loss)

    expected_grad = mx.zeros_like(x_data)
    assert mx.allclose(x.grad, expected_grad, atol=1e-3), f"Backward check failed. Expected {expected_grad}, got {x.grad}"


def test_dense(debug: bool=False):
    x = Tensor(mx.ones((1,2)), requires_grad=True, name='input')  # NHWC

    dense = Dense(2,2, activation='sigmoid')

    def tf_dense_fn(weights):
        return tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            use_bias=True, 
            kernel_initializer=tf.keras.initializers.Constant(weights['weights']), 
            bias_initializer=tf.keras.initializers.Constant(weights.get('biases', mx.zeros(2)))
        )

    compare_with_tf(
        custom_layer=dense,
        tf_layer_fn=tf_dense_fn,
        input_tensor=x,
        tf_param_map={
            'weights': lambda layer: layer.kernel,
            'biases': lambda layer: layer.bias,
        },
        debug=debug
    )


def test_activation(debug: bool=False):
    x = Tensor(mx.random.normal(shape=(1,10)), requires_grad=True, name='input')  # NHWC

    activation = Activation('relu')

    def tf_activation_fn(weights):
        return tf.keras.layers.Activation(
            activation='relu', 
        )

    compare_with_tf(
        custom_layer=activation,
        tf_layer_fn=tf_activation_fn,
        input_tensor=x,
        tf_param_map={
        },
        debug=debug
    )


def test_conv(debug: bool=False):
    x = Tensor(mx.ones((2, 5, 5, 3)), requires_grad=True, name='input')  # NHWC

    conv = Conv2D(in_channels=3, out_channels=2, kernel_size=3)

    def tf_conv_fn(weights):
        return tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding='valid',
            use_bias='biases' in weights,
            kernel_initializer=tf.constant_initializer(weights['weights']),
            bias_initializer=tf.constant_initializer(weights.get('biases', mx.zeros(2)))
        )

    compare_with_tf(
        custom_layer=conv,
        tf_layer_fn=tf_conv_fn,
        input_tensor=x,
        tf_param_map={
            'weights': lambda layer: layer.kernel,
            'biases': lambda layer: layer.bias,
        },
        debug=debug
    )


if __name__ == "__main__":
    # test_batchnorm_forward_backward()
    # test_dense(debug=True)
    # test_activation(debug=True)
    test_conv(debug=True)
    print("All tests passed!")