from tensorgraph.optimisers import SGD, Adam
from tensorgraph.tensor import Tensor
import tensorflow as tf
import mlx.core as mx
import pytest

def test_sgd_plain():
    w = Tensor([1.0])
    w.grad = mx.array([0.5])
    opt = SGD(learning_rate=0.1)
    opt.apply_gradients([w])
    assert abs(w.item() - 0.95) < 1e-6

def test_sgd_momentum():
    w = Tensor([1.0])
    w.grad = mx.array([0.5])
    opt = SGD(learning_rate=0.1, momentum=0.9)
    opt.apply_gradients([w])
    assert abs(w.item() - 0.95) < 1e-6
    w.grad = mx.array([0.5])
    opt.apply_gradients([w])
    # Expected after second step:
    # v1 = -0.05
    # v2 = 0.9 * (-0.05) - 0.05 = -0.095
    # w = 0.95 + (-0.095) = 0.855
    assert abs(w.item() - 0.855) < 1e-6

def test_sgd_nesterov():
    param = Tensor([1.0], requires_grad=True)
    param.grad = mx.array([0.5])

    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

    # First update
    opt.apply_gradients([param])
    assert abs(param.item() - 0.905) < 1e-6  # Expected: 0.905

    opt.apply_gradients([param])
    assert abs(param.item() - 0.7695) < 1e-6  # Expected: 0.7695


def quadratic_loss_and_grad(param: Tensor):
    diff = param.data - 5
    loss = diff * diff
    grad = 2 * diff
    return loss, grad

@pytest.mark.parametrize("learning_rate", [0.03, 0.05])
def test_adam_converges_to_target(learning_rate):
    param = Tensor(mx.array([0.0]))
    optimizer = Adam(learning_rate=learning_rate)

    for _ in range(500):  # Give enough steps for convergence
        loss, grad = quadratic_loss_and_grad(param)
        param.grad = grad
        optimizer.apply_gradients([param])

    # Final parameter should be close to 5
    assert abs(param.data.item() - 5) < 1e-2, f"Adam did not converge properly: {param.data.item()}, learning rate: {learning_rate}"


def test_adam_matches_tensorflow():
    # Config
    lr = 0.001
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    steps = 50

    # Our param
    param = Tensor(mx.array([0.0]))
    optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

    # TensorFlow param
    tf_param = tf.Variable([0.0], dtype=tf.float32)
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    for _ in range(steps):
        # Compute gradient manually for both
        grad_value = 2 * (param.data - 5)
        param.grad = grad_value
        optimizer.apply_gradients([param])

        with tf.GradientTape() as tape:
            loss = (tf_param - 5.0) ** 2
        grads = tape.gradient(loss, [tf_param])
        tf_optimizer.apply_gradients(zip(grads, [tf_param]))

    # Compare final values
    our_val = float(param.item())
    tf_val = float(tf_param.numpy().item())

    assert abs(our_val - tf_val) < 1e-6, f"Our Adam: {our_val}, TF Adam: {tf_val}"



if __name__ == "__main__":
    test_sgd_plain()
    test_sgd_momentum()
    test_sgd_nesterov()