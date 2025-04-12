import mlx.core as mx
import pytest
from tensor import Tensor

def test_min_forward():
    t = Tensor([[4.0, -1.0, 2.5, 0.0]])
    assert t.min().data.tolist() == [-1.0]

def test_max_forward():
    t = Tensor([[4.0, -1.0, 2.5, 0.0]])
    assert t.max().data.tolist() == [4.0]

def test_add_forward():
    a = Tensor([[2.0]])
    b = Tensor([[3.0]])
    c = a + b
    assert c.data.tolist() == [[5.0]]

def test_sub_forward():
    a = Tensor([[5.0]])
    b = Tensor([[3.0]])
    c = a - b
    assert c.data.tolist() == [[2.0]]

def test_mul_forward():
    a = Tensor([[2.0]])
    b = Tensor([[3.0]])
    c = a * b
    assert c.data.tolist() == [[6.0]]

def test_scalar_mul_forward():
    a = Tensor([[2.0]])
    c = a * 3.0
    assert c.data.tolist() == [[6.0]]

def test_div_forward():
    a = Tensor([[6.0]])
    b = Tensor([[2.0]])
    c = a / b
    assert c.data.tolist() == [[3.0]]

def test_pow_forward():
    a = Tensor([[2.0]])
    c = a ** 3
    assert c.data.tolist() == [[8.0]]

def test_matmul_forward():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0], [4.0]])
    c = a @ b
    assert c.data.tolist() == [[11.0]]

def test_sum_forward():
    a = Tensor([[1.0, 2.0, 3.0]])
    c = a.sum()
    assert c.data.tolist() == [6.0]

def test_mean_forward():
    a = Tensor([[1.0, 2.0, 3.0]])
    c = a.mean()
    assert c.data.tolist() == [2.0]

def test_relu_forward():
    a = Tensor([[-1.0, 0.0, 2.0]])
    c = a.relu()
    assert c.data.tolist() == [[0.0, 0.0, 2.0]]

def test_sigmoid_forward():
    a = Tensor([[0.0]])
    c = a.sigmoid()
    assert c.data.tolist() == [[0.5]]

def test_tanh_forward():
    a = Tensor([[0.0]])
    c = a.tanh()
    assert c.data.tolist() == [[0.0]]

def test_softmax_forward():
    t = Tensor(mx.array([[1.0, 2.0, 3.0]]), requires_grad=False)
    out = t.softmax()

    shifted = t.data - mx.min(t.data, axis=1)
    exp = mx.exp(shifted)
    expected = exp / mx.sum(exp, axis=1)

    assert out.data.shape == (1,3)
    # assert out.data.tolist() == expected.tolist()

def test_exp_forward():
    a = Tensor([[1.0]])
    c = a.exp()
    assert c.data.tolist() == mx.exp(mx.array([[1.0]])).tolist()

def test_log_forward():
    t = Tensor(mx.array([1.0, 10.0, 100.0]), requires_grad=False)
    log_t = t.log()
    expected = mx.log(t.data) / mx.log(mx.array(10))
    assert log_t.data.tolist() == expected.tolist()

def test_log_natural_forward():
    t = Tensor(mx.array([1.0, mx.e, mx.e**2]), requires_grad=False)
    log_t = t.log(base=mx.e)
    expected = mx.log(t.data)
    assert log_t.data.tolist() == expected.tolist()

def test_reshape_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.reshape(4)
    assert c.data.tolist() == mx.array([1.0, 2.0, 3.0, 4.0]).tolist()

def test_transpose_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.T()
    assert c.data.tolist() == mx.array([[1.0, 3.0], [2.0, 4.0]]).tolist()

def test_apply_sum_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.apply("sum")
    assert mx.allclose(c.data, mx.array([3.0, 7.0])).item()

def test_apply_mean_forward():
    a = Tensor([[1.0, 2.0], [3.0, 5.0]])
    c = a.apply("mean")
    assert mx.allclose(c.data, mx.array([1.5, 4.0])).item()

def test_apply_relu_forward():
    a = Tensor([[-1.0, 2.0], [0.0, -3.0]])
    c = a.apply("relu")
    assert mx.allclose(c.data, mx.array([[0.0, 2.0], [0.0, 0.0]])).item()

# def test_apply_softmax_forward():
#     data = mx.array([[1.0, 2.0, 3.0],
#                      [3.0, 0.0, -1.0]])
#     t = Tensor(data)
#     out = t.apply("softmax")

#     expected = mx.stack([mx.exp(row) / mx.sum(mx.exp(row)) for row in data])

#     assert out.data.shape == (2, 3)
#     assert mx.allclose(out.data, expected, rtol=1e-5)

# Backward tests (already existing)
def test_add_backward():
    a = Tensor([[2.0]], requires_grad=True)
    b = Tensor([[3.0]], requires_grad=True)
    c = a + b
    c.backward()
    assert a.grad.tolist() == [[1.0]]
    assert b.grad.tolist() == [[1.0]]

def test_sub_backward():
    a = Tensor([[5.0]], requires_grad=True)
    b = Tensor([[3.0]], requires_grad=True)
    c = a - b
    c.backward()
    assert a.grad.tolist() == [[1.0]]
    assert b.grad.tolist() == [[-1.0]]

def test_mul_backward():
    a = Tensor([[2.0]], requires_grad=True)
    b = Tensor([[3.0]], requires_grad=True)
    c = a * b
    c.backward()
    assert a.grad.tolist() == [[3.0]]
    assert b.grad.tolist() == [[2.0]]

def test_scalar_mul_backward():
    a = Tensor([[2.0]], requires_grad=True)
    c = a * 3.0
    c.backward()
    assert a.grad.tolist() == [[3.0]]

def test_div_backward():
    a = Tensor([[6.0]], requires_grad=True)
    b = Tensor([[2.0]], requires_grad=True)
    c = a / b
    c.backward()
    assert a.grad.tolist() == (1 / b.data).tolist()
    assert b.grad.tolist() == (-a.data / (b.data**2)).tolist()

def test_pow_backward():
    a = Tensor([[2.0]], requires_grad=True)
    c = a ** 3
    c.backward()
    assert a.grad.tolist() == ([[3 * (2.0 ** 2)]])

def test_matmul_backward():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0], [4.0]], requires_grad=True)
    c = a @ b
    c.backward()
    assert a.grad.tolist() == [[3.0, 4.0]]
    assert b.grad.tolist() == [[1.0], [2.0]]

def test_sum_backward():
    a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    c = a.sum()
    c.backward()
    assert a.grad.tolist() == [[1.0, 1.0, 1.0]]

def test_mean_backward():
    a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    c = a.mean()
    c.backward()
    assert a.grad.tolist() == mx.array([[1/3, 1/3, 1/3]]).tolist()

def test_relu_backward():
    a = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
    c = a.relu()
    c.backward(mx.array([1.0, 1.0, 1.0]))
    assert a.grad.tolist() == [[0.0, 0.0, 1.0]]

def test_sigmoid_backward():
    a = Tensor([[0.0]], requires_grad=True)
    c = a.sigmoid()
    c.backward()
    assert a.grad.tolist() == [[0.25]]

def test_tanh_backward():
    a = Tensor([[0.0]], requires_grad=True)
    c = a.tanh()
    c.backward()
    assert a.grad.tolist() == [[1.0]]

def test_softmax_backward():
    t = Tensor(mx.array([[1.0, 2.0, 3.0]]), requires_grad=True)
    out = t.softmax()

    assert mx.allclose(out.data, mx.array([0.0900, 0.2447, 0.6652]), atol=1e-4)

    # Simulate gradient from upstream (e.g. from loss)
    grad_out = mx.array([1.0, 0.0, 0.0])
    out.backward(grad_out)

    assert t.grad is not None
    assert t.grad.shape == t.data.shape
    # assert t.grad.tolist() == mx.array([[0.0819, -0.0220, -0.0599]]).tolist()

def test_exp_backward():
    a = Tensor([[1.0]], requires_grad=True)
    c = a.exp()
    c.backward()
    assert a.grad.tolist() == mx.exp(mx.array([[1.0]])).tolist()

def test_log_backward():
    t = Tensor(mx.array([[1.0, 10.0, 100.0]]), requires_grad=True)
    out = t.log()
    out.backward(mx.array([[1.0, 1.0, 1.0]]))
    expected_grad = 1 / (t.data * mx.log(mx.array(10)))
    assert t.grad.tolist() == expected_grad.tolist()

def test_log_natural_backward():
    t = Tensor(mx.array([[1.0, mx.e, mx.e**2]]), requires_grad=True)
    out = t.log(base=mx.e)
    out.backward(mx.ones_like(t.data))
    expected_grad = 1 / t.data
    assert t.grad.tolist() == expected_grad.tolist()

def test_reshape_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c = a.reshape(4)
    c = c.sum()
    c.backward()
    assert a.grad.tolist() == mx.array([[1.0, 1.0], [1.0, 1.0]]).tolist()

def test_transpose_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c = a.T()
    c = c.sum()
    c.backward()
    assert a.grad.tolist() == mx.array([[1.0, 1.0], [1.0, 1.0]]).tolist()

def test_apply_sum_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c = a.apply("sum")
    c.backward(mx.array([1.0, 1.0]))  # dL/dc = [1, 1]
    expected = mx.array([[1.0, 1.0], [1.0, 1.0]])
    assert mx.allclose(a.grad, expected).item()

def test_apply_mean_backward():
    a = Tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
    c = a.apply("mean")
    c.backward(mx.array([1.0, 1.0]))
    expected = mx.array([[0.5, 0.5], [0.5, 0.5]])
    assert mx.allclose(a.grad, expected).item()

def test_apply_relu_backward():
    a = Tensor([[-1.0, 2.0], [0.0, -3.0]], requires_grad=True)
    c = a.apply("relu")
    c.backward(mx.ones_like(c.data))
    expected = mx.array([[0.0, 1.0], [0.0, 0.0]])
    assert mx.allclose(a.grad, expected).item()

# def test_apply_softmax_backward():
#     data = mx.array([[1.0, 2.0, 3.0],
#                      [3.0, 0.0, -1.0]])
#     t = Tensor(data, requires_grad=True)
#     out = t.apply("softmax")

#     # Dummy gradient from upstream (same shape as out)
#     grad_out = mx.ones_like(out.data)
#     out.backward(grad_out)

#     assert t.grad is not None
#     assert t.grad.shape == data.shape

if __name__ == '__main__':
    test_mean_backward()