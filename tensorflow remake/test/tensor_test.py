import mlx.core as mx
import pytest
from tensorgraph.losses import Tensor
from tensorgraph.visualiser import visualize_tensor_graph
from tensorgraph.layers import resolve_initializer

def test_forward_add():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[5, 6], [7, 8]], requires_grad=True)
    c = a + b
    assert (c.data == mx.array([[6, 8], [10, 12]])).all()

def test_forward_mul():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[2, 3], [4, 5]], requires_grad=True)
    c = a * b
    assert (c.data == mx.array([[2, 6], [12, 20]])).all()

def test_forward_matmul():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], requires_grad=True)
    c = a @ b
    assert (c.data[0] == mx.array([[1, 2], [3, 4]])).all()
    assert (c.data[1] == mx.array([[10, 12], [14, 16]])).all()

def test_sum_all_elements():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.sum()  # scalar
    c.backward()
    expected = mx.ones_like(a.data)
    assert mx.allclose(a.grad, expected), f"Sum over all elements failed, expected {expected}, got {a.grad}"

def test_sum_axis_0():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.sum(axis=0)
    c.backward(mx.ones_like(c.data))
    expected = mx.array([[1.0, 1.0], [1.0, 1.0]])
    assert mx.allclose(a.grad, expected), f"Sum over axis=0 failed, expected {expected}, got {a.grad}"

def test_sum_axis_1_keepdims():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.sum(axis=1, keepdims=True)  # shape (2,1)
    c.backward(mx.ones_like(c.data))
    expected = mx.array([[1.0, 1.0], [1.0, 1.0]])
    assert mx.allclose(a.grad, expected), f"Sum over axis=1 with keepdims failed, expected {expected}, got {a.grad}"

def test_mean_all_elements():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.mean(axis=None)  # scalar
    c.backward()
    expected = mx.ones_like(a.data) / a.data.size
    assert mx.allclose(a.grad, expected), f"Mean over all elements failed, expected {expected}, got {a.grad}"

def test_mean_axis_0():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.mean(axis=0)
    c.backward(mx.ones_like(c.data))  # shape (2,)
    expected = mx.array([[0.5, 0.5], [0.5, 0.5]])
    assert mx.allclose(a.grad, expected), f"Mean over axis=0 failed, expected {expected}, got {a.grad}"

def test_mean_axis_1_keepdims():
    a = Tensor(mx.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.mean(axis=1, keepdims=True)  # shape (2,1)
    c.backward(mx.ones_like(c.data))
    expected = mx.array([[0.5, 0.5], [0.5, 0.5]])
    assert mx.allclose(a.grad, expected), f"Mean over axis=1 with keepdims failed, expected {expected}, got {a.grad}"


def test_forward_relu_sigmoid_softmax():
    a = Tensor([[-1, 0, 1], [2, -2, 0]], requires_grad=True)
    relu = a.relu()
    sigmoid = a.sigmoid()
    softmax = a.softmax()
    assert (relu.data == mx.array([[0, 0, 1], [2, 0, 0]])).all()
    assert (sigmoid.data >= 0).all() and (sigmoid.data <= 1).all()
    assert softmax.data.shape == a.data.shape
    assert (mx.abs(softmax.data.sum(axis=-1) - 1) < 1e-6).all()

def test_backward_add():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[5, 6], [7, 8]], requires_grad=True)
    c = a + b
    c.backward(mx.array([[1, 1], [1, 1]]))
    assert (a.grad == mx.array([[1, 1], [1, 1]])).all()
    assert (b.grad == mx.array([[1, 1], [1, 1]])).all()

def test_backward_mul():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[2, 3], [4, 5]], requires_grad=True)
    c = a * b
    c.backward(mx.array([[1, 1], [1, 1]]))
    assert (a.grad == b.data).all()
    assert (b.grad == a.data).all()

def test_backward_matmul():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], requires_grad=True)
    c = a @ b
    c.backward(mx.ones_like(c.data))
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape

def test_backward_relu():
    a = Tensor([[-1, 0, 1], [2, -2, 0]], requires_grad=True)
    relu = a.relu()
    relu.backward(mx.array([[1, 1, 1], [1, 1, 1]]))
    expected = mx.array([[0, 0, 1], [1, 0, 0]])
    assert (a.grad == expected).all()

def test_backward_sigmoid():
    a = Tensor([[0.0, 2.0], [-1.0, 1.0]], requires_grad=True)
    s = a.sigmoid()
    s.backward(mx.ones_like(s.data))
    # Check shape and grad range
    assert a.grad.shape == a.data.shape
    assert (a.grad >= 0).all() and (a.grad <= 1).all()

def test_forward_getitem():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = a[1]
    assert (b.data == mx.array([3, 4])).all()

def test_backward_getitem():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = a[1]
    b.backward(mx.array([1, 1]))
    assert (a.grad == mx.array([[0, 0], [1, 1]])).all()


def test_forward_min_max():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    min_val = a.min()
    max_val = a.max()
    expected_min = mx.array([[1]], dtype=min_val.data.dtype)
    expected_max = mx.array([[4]], dtype=max_val.data.dtype)
    assert min_val.data.shape == expected_min.shape
    assert max_val.data.shape == expected_max.shape
    assert (min_val.data == expected_min).all()
    assert (max_val.data == expected_max).all()

def test_backward_min_max():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    min_val = a.min()
    max_val = a.max()
    min_val.backward(mx.array([[1], [1]]))
    max_val.backward(mx.array([[1], [1]]))
    # Grad for min/max is not implemented, so grad should be None
    assert a.grad is None


def test_forward_sub():
    a = Tensor([[5, 6], [7, 8]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4]], requires_grad=True)
    c = a - b
    assert (c.data == mx.array([[4, 4], [4, 4]])).all()

def test_backward_sub():
    a = Tensor([[5, 6], [7, 8]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4]], requires_grad=True)
    c = a - b
    c.backward(mx.array([[1, 1], [1, 1]]))
    assert (a.grad == mx.array([[1, 1], [1, 1]])).all()
    assert (b.grad == mx.array([[-1, -1], [-1, -1]])).all()


def test_forward_mul_rmul():
    a = Tensor([[2, 3], [4, 5]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4]], requires_grad=True)
    c = a * b
    d = 2 * a
    assert (c.data == mx.array([[2, 6], [12, 20]])).all()
    assert (d.data == mx.array([[4, 6], [8, 10]])).all()

def test_backward_mul_rmul():
    a = Tensor([[2, 3], [4, 5]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4]], requires_grad=True)
    c = a * b
    c.backward(mx.array([[1, 1], [1, 1]]))
    assert (a.grad == b.data).all()
    assert (b.grad == a.data).all()
    a2 = Tensor([[2, 3], [4, 5]], requires_grad=True)
    d = 2 * a2
    d.backward(mx.array([[1, 1], [1, 1]]))
    assert (a2.grad == mx.array([[2, 2], [2, 2]])).all()


def test_forward_neg():
    a = Tensor([[1, -2], [3, -4]], requires_grad=True)
    b = -a
    assert (b.data == mx.array([[-1, 2], [-3, 4]])).all()

def test_backward_neg():
    a = Tensor([[1, -2], [3, -4]], requires_grad=True)
    b = -a
    b.backward(mx.array([[1, 1], [1, 1]]))
    assert (a.grad == mx.array([[-1, -1], [-1, -1]])).all()


def test_forward_div():
    a = Tensor([[4, 9], [16, 25]], requires_grad=True)
    b = Tensor([[2, 3], [4, 5]], requires_grad=True)
    c = a / b
    assert (c.data == mx.array([[2, 3], [4, 5]])).all()

def test_backward_div():
    a = Tensor([[4, 9], [16, 25]], requires_grad=True)
    b = Tensor([[2, 3], [4, 5]], requires_grad=True)
    c = a / b
    c.backward(mx.array([[1, 1], [1, 1]]))
    assert mx.allclose(a.grad, mx.array([[0.5, 0.33333334], [0.25, 0.2]]))
    assert mx.allclose(b.grad, mx.array([[-1, -1], [-1, -1]]))

def test_forward_tanh_exp_log_step():
    a = Tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    tanh = a.tanh()
    exp = a.exp()
    log = Tensor([[1.0, 10.0], [100.0, 1000.0]], requires_grad=True).log(base=10.0)
    step = a.step()
    assert tanh.data.shape == a.data.shape
    assert exp.data.shape == a.data.shape
    assert log.data.shape == (2, 2)
    assert (step.data == mx.array([[0, 1], [0, 1]])).all()

def test_backward_tanh_exp_log_step():
    a = Tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    tanh = a.tanh()
    tanh.backward(mx.ones_like(tanh.data))
    assert a.grad is not None and a.grad.shape == a.data.shape
    a2 = Tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    exp = a2.exp()
    exp.backward(mx.ones_like(exp.data))
    assert a2.grad is not None and a2.grad.shape == a2.data.shape
    a3 = Tensor([[1.0, 10.0], [100.0, 1000.0]], requires_grad=True)
    log = a3.log(base=10.0)
    log.backward(mx.ones_like(log.data))
    assert a3.grad is not None and a3.grad.shape == a3.data.shape
    a4 = Tensor([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    step = a4.step()
    step.backward(mx.ones_like(step.data))
    assert a4.grad is not None and a4.grad.shape == a4.data.shape


def test_forward_reshape_transpose():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    reshaped = a.reshape(2,2,1)
    transposed = a.T()
    assert reshaped.data.shape == (2,2,1)
    assert transposed.data.shape == (2,2)
    assert (transposed.data == mx.array([[1, 3], [2, 4]])).all()

def test_backward_reshape_transpose():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    reshaped = a.reshape(2,2,1)
    reshaped.backward(mx.ones_like(reshaped.data))
    assert a.grad is not None and a.grad.shape == a.data.shape
    a2 = Tensor([[1, 2], [3, 4]], requires_grad=True)
    transposed = a2.T()
    transposed.backward(mx.ones_like(transposed.data))
    assert a2.grad is not None and a2.grad.shape == a2.data.shape


def test_broadcast_add():
    a = Tensor([[1], [2]], requires_grad=True)  # shape (2, 1)
    b = Tensor([[10, 20]], requires_grad=True)  # shape (1, 2)
    c = a + b  # shape (2, 2)
    assert (c.data == mx.array([[11, 21], [12, 22]])).all()
    c.backward(mx.ones_like(c.data))
    assert (a.grad == mx.array([[2], [2]])).all()
    assert (b.grad == mx.array([[2, 2]])).all()

def test_scalar_mul():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    c = a * 3
    assert (c.data == mx.array([[3, 6], [9, 12]])).all()
    c.backward(mx.ones_like(c.data))
    assert (a.grad == mx.array([[3, 3], [3, 3]])).all()

def test_zero_dim_tensor():
    a = Tensor(5.0, requires_grad=True)
    b = a * 2
    assert b.data == 10.0
    b.backward(mx.array(1.0))
    assert a.grad == 2.0

def test_gradient_accumulation():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = a * 2
    b.backward(mx.ones_like(b.data))
    b.backward(mx.ones_like(b.data))
    assert (a.grad == mx.array([[4.0, 4.0]])).all()  # Should accumulate

def test_no_grad_flag():
    a = Tensor([[1, 2]], requires_grad=False)
    b = a * 2
    b.backward(mx.ones_like(b.data))
    assert a.grad is None

def test_chained_operations():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)
    c = (a + b) * 2
    c.backward(mx.ones_like(c.data))
    assert (a.grad == mx.array([[2.0, 2.0]])).all()
    assert (b.grad == mx.array([[2.0, 2.0]])).all()

def test_sum_axis_variants():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    s0 = a.sum(axis=0, keepdims=True)
    s1 = a.sum(axis=1, keepdims=True)
    assert (s0.data == mx.array([[4, 6]])).all()
    assert (s1.data == mx.array([[3], [7]])).all()
    s0.backward(mx.ones_like(s0.data))
    assert (a.grad == mx.ones_like(a.data)).all()

def test_masked_operations():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    mask = Tensor([[1, 0], [0, 1]], requires_grad=False)
    masked = a * mask
    assert (masked.data == mx.array([[1, 0], [0, 4]])).all()
    masked.backward(mx.ones_like(masked.data))
    assert (a.grad == mx.array([[1, 0], [0, 1]])).all()

def test_stack_forward_backward():
    a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    b = Tensor([[5., 6.], [7., 8.]], requires_grad=True)
    c = a + b  # simple addition to create a dependency

    # Forward pass: Stack along new axis 0
    s = Tensor.stack([a, b, c], axis=0)  # shape: (2, 2, 2)

    # Check forward output
    expected_data = mx.array([[[1., 2.], [3., 4.]],
                              [[5., 6.], [7., 8.]],
                              [[6., 8.], [10., 12.]]])
    assert (s.data == expected_data).all(), "Forward stack failed"

    # Create a simple scalar loss and backward
    loss = s.sum()  # dL/ds = 1 for each element
    loss.backward()

    visualize_tensor_graph(loss)

    # Check gradients: all should be 1
    excepted_grad_ab = resolve_initializer('constant', a.shape, 2.0)
    expected_grad_c = mx.ones_like(a.data)
    assert (a.grad == excepted_grad_ab).all(), f"Gradient w.r.t a incorrect, expected {excepted_grad_ab}, got {a.grad}"
    assert (b.grad == excepted_grad_ab).all(), f"Gradient w.r.t b incorrect, expected {excepted_grad_ab}, got {b.grad}"
    assert (c.grad == expected_grad_c).all(), f"Gradient w.r.t c incorrect, expected {expected_grad_c}, got {c.grad}"

    print("test_stack_forward_backward passed ✅")

def test_eq_operation():
    x = Tensor([1, 2, 3])
    y = Tensor([1, 4, 3])
    
    mask = x == y  # Should return Tensor with boolean values
    expected = mx.array([True, False, True])

    assert (mask.data == expected).all(), f"Expected {expected}, got {mask.data}"
    assert mask._op == "eq"
    assert mask._parents == [x, y]

def test_masked_select_all_true():
    x = Tensor([10, 20, 30])
    mask = Tensor([True, True, True])

    selected = x.masked_select(mask)
    expected = mx.array([10, 20, 30])

    assert (selected.data == expected).all(), f"Expected {expected}, got {selected.data}"
    assert selected._op == "masked_select"
    assert selected._parents == [x, mask]

def test_masked_select_partial():
    x = Tensor([5, 6, 7, 8])
    mask = Tensor([True, False, True, False])

    selected = x.masked_select(mask)
    expected = mx.array([5, 0, 7, 0])  # if zeros for masked-out positions

    assert (selected.data == expected).all(), f"Expected {expected}, got {selected.data}"

def test_masked_select_with_eq_mask():
    x = Tensor([1, 2, 3])
    y = Tensor([1, 4, 3])
    
    mask = x == y
    selected = x.masked_select(mask)
    expected = mx.array([1, 0, 3])  # zeros where mask is False

    assert (selected.data == expected).all()

def test_and_operation():
    a = Tensor([True, False, True])
    b = Tensor([True, True, False])

    result = a & b
    expected = mx.array([True, False, False])

    assert (result.data == expected).all(), f"Expected {expected}, got {result.data}"
    assert result._op == "and"
    assert result._parents == [a, b]

def test_or_operation():
    a = Tensor([True, False, True])
    b = Tensor([False, False, True])

    result = a | b
    expected = mx.array([True, False, True])

    assert (result.data == expected).all(), f"Expected {expected}, got {result.data}"
    assert result._op == "or"
    assert result._parents == [a, b]

def test_invert_operation():
    a = Tensor([True, False, True])

    result = ~a
    expected = mx.array([False, True, False])

    assert (result.data == expected).all(), f"Expected {expected}, got {result.data}"
    assert result._op == "invert"
    assert result._parents == [a]

def test_combined_logic():
    a = Tensor([True, False, True])
    b = Tensor([True, True, False])

    result = (a & b) | ~b
    expected = mx.array([True, False, True])  # AND: [True, False, False], NOT b: [False, False, True], OR: [True, False, True]

    assert (result.data == expected).all(), f"Expected {expected}, got {result.data}"

