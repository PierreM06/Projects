import mlx.core as mx
import numpy as np
from typing import Callable, Optional, Any, Union

Axis = Optional[Union[int, tuple[int, ...]]]
Shape = tuple[int, ...]

# Type hint for reduction functions
ReductionFn = Callable[['Tensor', Axis, bool], 'Tensor']

def unbroadcast(grad: Any, original_shape: Any) -> Any:
    # Add leading dims to match grad shape
    diff = len(grad.shape) - len(original_shape)
    target_shape = (1,) * diff + original_shape
    for axis, (g, o) in enumerate(zip(grad.shape, target_shape)):
        if o == 1 and g != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    if diff > 0:
        grad = grad.reshape(original_shape)
    return grad

class Tensor:
    def __init__(self, data: Any, requires_grad: bool = False, name: Optional[str] = None) -> None:
        arr = mx.array(data, dtype=mx.float32)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1))

        self.data: mx.array = arr
        self.requires_grad: bool = requires_grad
        self.grad: Optional[mx.array] = None
        self._parents: list['Tensor'] = []
        self._op: Optional[str] = None
        self._name: Optional[str] = name
        self._extra: dict[str, Any] = {}
        self._parent_grads: dict[int, mx.array] = {}

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def shape(self) -> Shape:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    def item(self) -> Any:
        if self.size != 1:
            return self.data.sum().item()
        return self.data.item()
    
    def set_name(self, name: str) -> None:
        """
        Set a name for the tensor, useful for debugging and visualization.
        """
        self._name = name
    
    def zero_grad(self) -> None:
        self.grad = None
        self._op = None

    def update(self, value: 'Tensor | mx.array') -> None:
        if isinstance(value, Tensor):
            self.data -= value.data
        elif isinstance(value, mx.array):
            self.data -= value
        else:
            raise ValueError(f'Unsuported type for updating tensor data, {type(value)} is not supported')

    def detach(self) -> 'Tensor':
        return Tensor(self.data, requires_grad=False)
    
    def tolist(self) -> list:
        """
        Converts the tensor data to a nested list.
        """
        return self.data.tolist()
    
    def numpy(self) -> np.ndarray:
        """
        Converts the tensor data to a NumPy array.
        """
        return np.array(self.data)
    
    def __iter__(self):
        # Iterate over batch
        for i in range(self.data.shape[0]):
            yield Tensor(self.data[i], requires_grad=self.requires_grad)
    
# ============================================================
# Comparisons
# ============================================================
    def __eq__(self, other: 'Tensor') -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.equal(self.data, other_data.data))
        result._parents=[self, other]
        result._op="eq"
        return result
    
    def __gt__(self, other) -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.greater(self.data, other_data.data))
        result._parents=[self, other]
        result._op="gt"
        return result

    def __lt__(self, other) -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.less(self.data, other_data.data))
        result._parents=[self, other]
        result._op="lt"
        return result
    
    def __ne__(self, other: 'Tensor') -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.not_equal(self.data, other_data.data))
        result._parents=[self, other]
        result._op="eq"
        return result
    
    def __and__(self, other) -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.logical_and(self.data, other_data.data))
        result._parents=[self, other]
        result._op="and"
        return result

    def __or__(self, other) -> 'Tensor':
        other_data = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(mx.logical_or(self.data, other_data.data))
        result._parents=[self, other]
        result._op="or"
        return result
    
    def __invert__(self):
        result = Tensor(mx.logical_not(self.data))
        result._parents=[self]
        result._op="invert"
        return result
    
# ============================================================
# Graph breaking
# ============================================================
    def masked_select(self, mask: 'Tensor') -> 'Tensor':
        if mask.data.shape != self.data.shape:
            raise ValueError("Mask shape must match tensor shape")
        # if mask.data.dtype != mx.bool:
        #     raise ValueError("Mask must be of dtype bool")

        # Convert mask to numpy and extract flat indices where it's True
        flat_mask = np.array(mask.data.flatten())
        indices_np = np.nonzero(flat_mask)[0].astype(np.int32)
        if indices_np.size == 0:
            return Tensor(mx.array([], dtype=self.data.dtype))

        indices = mx.array(indices_np, dtype=mx.int32)

        # Flatten input tensor and take matching values
        flat_data = self.data.flatten()
        selected = mx.take(flat_data, indices)

        result = Tensor(selected)
        result._parents = [self, mask]
        result._op = "masked_select"
        return result

    def __getitem__(self, key: 'Union[int, slice, Tensor, mx.array]') -> 'Tensor':
        # Boolean mask indexing
        if isinstance(key, Tensor):
            return self.masked_select(key)

        # Fallback to normal slicing/indexing
        # Batch-aware slicing
        result = Tensor(self.data[key], requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'getitem'
        result._extra['idx'] = key
        return result

    def __setitem__(self, idx: Union[int, slice], value: Any) -> None:
        # Batch-aware setting
        if not isinstance(value, Tensor):
            value = Tensor(value, requires_grad=self.requires_grad)
        self.data[idx] = value.data

# ============================================================
# Operations
# ============================================================
    def _binary_op(self, other: Any, op_name: str, op_fn: Callable[[mx.array, mx.array], mx.array]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # Broadcast batch dimension if needed
        a, b = self.data, other.data
        # if a.shape[0] != b.shape[0]:
        #     if a.shape[0] == 1:
        #         a = mx.broadcast_to(a, (b.shape[0],) + a.shape[1:])
        #     elif b.shape[0] == 1:
        #         b = mx.broadcast_to(b, (a.shape[0],) + b.shape[1:])
        result = Tensor(op_fn(a, b), requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = op_name
        return result

    def __add__(self, other: Any) -> 'Tensor':
        return self._binary_op(other, 'add', lambda a, b: a + b)

    def __sub__(self, other: Any) -> 'Tensor':
        return self._binary_op(other, 'sub', lambda a, b: a - b)

    def __mul__(self, other: Any) -> 'Tensor':
        return self._binary_op(other, 'mul', lambda a, b: a * b)

    def __rmul__(self, other: Any) -> 'Tensor':
        return self * other
    
    def __truediv__(self, other: Any) -> 'Tensor':
        return self._binary_op(other, 'div', lambda a, b: a / b)
    
    def __pow__(self, exponent: Any) -> 'Tensor':
        return self._binary_op(exponent, 'pow', lambda a, b: a ** b)

    def __neg__(self) -> 'Tensor':
        result = Tensor(-self.data, requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'neg'
        return result
    
    def __matmul__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # Batch matmul: if both have batch, do per-batch matmul
        a, b = self.data, other.data
        if a.shape[0] == b.shape[-1] or a.shape[-1] == b.shape[0]:
            result_data = mx.matmul(a, b)
        elif a.shape[0] == 1:
            result_data = mx.matmul(mx.broadcast_to(a, b.shape), b)
        elif b.shape[0] == 1:
            result_data = mx.matmul(a, mx.broadcast_to(b, a.shape))
        else:
            raise ValueError("Batch sizes do not match for matmul")
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'matmul'
        return result
    
    def max(self, axis: Axis=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.max(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'max'
        return result
    
    def argmax(self, axis: Axis=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.argmax(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'argmax'
        return result
    
    def min(self, axis: Axis=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.min(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'min'
        return result
    
    def argmin(self, axis: Axis=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.argmin(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'argmin'
        return result
    
    def sum(self, axis: Optional[int]=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'sum'
        return result
    
    def mean(self, axis: Optional[int]=None, keepdims: bool=True) -> 'Tensor':
        result = Tensor(mx.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'mean'
        result._extra['axis'] = axis
        result._extra['keepdims'] = keepdims
        return result
    
    def relu(self) -> 'Tensor':
        result = Tensor(mx.maximum(self.data, 0), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'relu'
        return result
    
    def sigmoid(self) -> 'Tensor':
        result = Tensor(mx.sigmoid(self.data), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'sigmoid'
        return result
    
    def tanh(self) -> 'Tensor':
        result = Tensor(mx.tanh(self.data), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'tanh'
        return result
    
    def softmax(self) -> 'Tensor':
        # Numerically stable softmax over the last axis
        shifted = self - self.max(axis=-1, keepdims=True)
        exps = shifted.exp()
        summed = exps.sum(axis=-1, keepdims=True)
        result = exps / summed
        result._name = 'softmax'
        result._extra['op'] = 'softmax'
        return result
    
    def exp(self) -> 'Tensor':
        result = Tensor(mx.exp(self.data), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'exp'
        return result
    
    def log(self, base: complex = 10.0) -> 'Tensor':
        if base == mx.e:
            data = mx.log(self.data)
        else:
            data = mx.log(self.data) / mx.log(mx.array(base))
        result = Tensor(data, requires_grad=self.requires_grad)
        result._parents = [self, Tensor(base)]
        result._op = "log"
        return result
    
    def step(self) -> 'Tensor':
        out = Tensor((self.data > 0).astype(self.data.dtype), requires_grad=self.requires_grad)
        out._parents = [self]
        out._op = 'step'
        return out
    
# ============================================================
# Functional operations
# ============================================================
    def reshape(self, *shape: Any) -> 'Tensor':
        # Support shape passed as a single tuple or multiple arguments
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        result = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'reshape'
        return result

    def T(self, axis: Optional[tuple[int, ...]] = None) -> 'Tensor':
        """
        Transpose tensor over given axes, batch-aware.
        If axis is None, transpose last two axes for each batch.
        Batch axis (axis 0) is never moved.
        """
        if axis is None:
            if self.data.ndim < 3:
                data = mx.swapaxes(self.data, -2, -1)
                axes_used = (-2, -1)  # Implicit case
            else:
                # Batch-aware transpose of last two axes
                axes_used = (0,) + tuple(range(1, self.data.ndim - 2)) + (self.data.ndim - 1, self.data.ndim - 2)
                data = mx.transpose(self.data, axes_used)
        else:
            if len(axis) == len(self.shape):
                axes_used = tuple(axis)
            else:
                axes_used = (0,) + tuple(axis)
            data = mx.transpose(self.data, axes_used)

        result = Tensor(data, requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'transpose'
        result._extra['axis'] = axes_used
        return result
    
    @staticmethod
    def stack(tensors: list['Tensor'], axis: int = 0) -> "Tensor":
        assert len(tensors) > 0, "Cannot stack an empty list of tensors."
        
        shape = tensors[0].shape
        for t in tensors:
            assert isinstance(t, Tensor), "All items must be Tensor instances."
            assert t.shape == shape, "All tensors must have the same shape to stack."

        requires_grad = any(t.requires_grad for t in tensors)

        data = [t.data for t in tensors]
        stacked = mx.stack(data, axis=axis)
        result = Tensor(stacked, requires_grad=requires_grad)
        result._parents = tensors
        result._op = 'stack'
        result._extra['axis'] = axis
        return result
    
# ============================================================
# Gradients
# ============================================================
    def backward(self, grad: Optional[Any] = None) -> None:
        if not self.requires_grad:
            return
        if grad is None:
            grad = mx.ones_like(self.data)

        if self.grad is None:
            self.grad = mx.array(grad)
        else:
            self.grad += grad

        if self._op == 'stack':
            axis = self._extra['axis']
            parents = self._parents  # list of input tensors

            # Use MLX's split function to split grad_output along axis into len(parents) pieces
            grad_slices = mx.split(grad, indices_or_sections=len(parents), axis=axis)

            # Now set the gradient for each parent tensor
            for parent, grad_slice in zip(parents, grad_slices):
                if parent.requires_grad:
                    parent.backward(grad_slice)

            return


        for parent in self._parents:
            if not parent.requires_grad:
                continue

            grad_to_pass = None
            
            if self._op == 'add':
                grad_to_pass = grad
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'sub':
                grad_to_pass = grad if parent is self._parents[0] else -grad
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad_to_pass, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'neg':
                grad_to_pass = -grad
                parent.backward(grad_to_pass)
            elif self._op == 'mul':
                other = self._parents[1] if parent is self._parents[0] else self._parents[0]
                grad_to_pass = grad * other.data
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad_to_pass, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'div':
                grad_to_pass = grad/self._parents[1].data if parent is self._parents[0] else (-self._parents[0].data * grad) / (self._parents[1].data**2)
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad_to_pass, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'pow':
                grad_to_pass = grad * self._parents[1].data * (parent.data ** (self._parents[1].data - 1))
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad_to_pass, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'matmul':
                if parent is self._parents[0]:
                    grad_to_pass = mx.matmul(grad, self._parents[1].data.swapaxes(-2, -1))
                else:
                    grad_to_pass = mx.matmul(self._parents[0].data.swapaxes(-2, -1), grad)
                if self.shape != parent.shape:
                    grad_to_pass = unbroadcast(grad_to_pass, parent.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'sum':
                grad_to_pass = mx.ones_like(parent.data) * grad
                parent.backward(grad_to_pass)
            elif self._op == 'mean':
                axis = self._extra['axis']
                keepdims = self._extra['keepdims']
                shape = parent.data.shape
                if axis is None:
                    count = parent.data.size
                    grad_to_pass = mx.ones_like(parent.data) * grad / count
                else:
                    # Normalize axis to tuple
                    if isinstance(axis, int):
                        axis = (axis,)
                    
                    # Compute number of elements reduced
                    count = 1
                    for ax in axis:
                        count *= shape[ax]

                    # Expand grad dims if needed
                    if not keepdims:
                        for ax in sorted(axis):
                            grad = mx.expand_dims(grad, ax)

                    grad_to_pass = mx.broadcast_to(grad, shape) / count

                parent.backward(grad_to_pass)
            elif self._op == 'relu':
                grad_to_pass = grad * (self.data > 0)
                parent.backward(grad_to_pass)
            elif self._op == 'sigmoid':
                grad_to_pass = grad * self.data * (1 - self.data)
                parent.backward(grad_to_pass)
            elif self._op == 'tanh':
                grad_to_pass = grad * (1 - mx.tanh(parent.data)**2)
                parent.backward(grad_to_pass)
            elif self._op == 'step':
                grad_to_pass = mx.zeros_like(self.data)
                parent.backward(grad_to_pass)
            elif self._op == 'exp':
                grad_to_pass = grad * mx.exp(parent.data)
                parent.backward(grad_to_pass)
            elif self._op == 'log':
                x, base = self._parents
                if parent is x:
                    if base == mx.e:
                        grad_to_pass = grad / parent.data
                    else:
                        grad_to_pass = grad / (parent.data * mx.log(mx.array(base.data)))
                    parent.backward(grad_to_pass)
                elif parent is base:
                    grad_to_pass = -grad * mx.log(x.data) / (mx.log(base.data) ** 2 * base.data)
                    parent.backward(grad_to_pass)
            elif self._op == 'reshape':
                grad_to_pass = grad.reshape(parent.data.shape)
                parent.backward(grad_to_pass)
            elif self._op == 'transpose':
                axes = self._extra['axis']
                inverse_axes = [0] * len(axes)
                for i, a in enumerate(axes):
                    inverse_axes[a] = i
                grad_to_pass = mx.transpose(grad, inverse_axes)
                parent.backward(grad_to_pass)
            elif self._op == 'getitem':
                grad_to_pass = mx.zeros_like(parent.data)
                grad_to_pass[self._extra['idx']] = grad
                parent.backward(grad_to_pass)

            self._parent_grads[id(parent)] = grad_to_pass
