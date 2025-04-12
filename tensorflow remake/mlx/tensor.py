import mlx.core as mx

def batch_aware_reduction(fn):
    def wrapped(self, axis=None, keepdims=False):
        if axis is None:
            if self.data.ndim <= 1:
                axis = None  # scalar or no batch
            else:
                axis = tuple(range(1, self.data.ndim))  # exclude batch dim

        return fn(self, axis=axis, keepdims=keepdims)
    return wrapped


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data: mx.array = mx.array(data)  # Store the MLX array
        self.requires_grad = requires_grad  # Track if gradients are needed
        self.grad: mx.array = None  #type: ignore Gradient storage
        self._parents: list[Tensor] = []  # Track dependencies for computation graph
        self._op: str = None  #type: ignore Operation that created this tensor
        self._idx: int = None  #type:ignore
        self._extra: str = None  #type:ignore

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self.data.shape
    
    def apply(self, method: str):
        """
        Applies a Tensor method (like 'sum', 'mean', 'relu', etc.) to each element
        along the first dimension (like a map over rows in a 2D tensor).

        Example:
            a.apply("sum") applies a.sum() on each row/sub-tensor.
        """
        if self.data.ndim == 0:
            raise ValueError("Cannot apply method on scalar Tensor.")
        elif self.data.ndim == 1:
            return getattr(self, method)()

        results = []
        for i in range(self.data.shape[0]):
            sub_tensor = self[i]
            result = getattr(sub_tensor, method)()
            result._parents = [sub_tensor]
            results.append(result)

        result = Tensor([r.data for r in results], requires_grad=self.requires_grad)
        # results.insert(0, self)
        result._parents = results
        result._op = 'apply'

        return result
    
    def zero_grad(self):
        self.grad = None #type:ignore

    def detach(self):
        return Tensor(self.data, requires_grad=False)
    
    def __iter__(self):
        for i in range(len(self.data)):
            yield Tensor(self.data[i], requires_grad=self.requires_grad)

    def __getitem__(self, idx):
        # Slice the data
        sliced_data = self.data[idx]
        # Wrap in Tensor and preserve requires_grad
        result = Tensor(sliced_data, requires_grad=self.requires_grad)
        
        # Track parent for autodiff
        result._parents = [self]
        result._op = 'getitem'
        result._idx = idx  # Save the index used (important for backward)
        
        return result

    @batch_aware_reduction
    def max(self, axis=None, keepdims=False):
        result = Tensor(mx.max(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'max'
        return result
    
    @batch_aware_reduction
    def min(self, axis=None, keepdims=False):
        result = Tensor(mx.min(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
        result._parents = [self]
        result._op = 'min'
        return result

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'add'
        return result
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'sub'
        return result
    
    def __neg__(self):
        result = Tensor(self.data * -1, requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'neg'
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'mul'
        return result
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'div'
        return result
    
    def __pow__(self, exponent):
        result = Tensor(self.data ** exponent, requires_grad=self.requires_grad)
        result._parents = [self, Tensor(exponent)]
        result._op = 'pow'
        return result
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(mx.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        result._parents = [self, other]
        result._op = 'matmul'
        return result
    
    @batch_aware_reduction
    def sum(self, axis=None, keepdims=False):
        result = Tensor(mx.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'sum'
        return result
    
    @batch_aware_reduction
    def mean(self, axis=None, keepdims=False):
        result = Tensor(mx.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'mean'
        return result
    
    def relu(self):
        result = Tensor(mx.maximum(self.data, 0), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'relu'
        return result
    
    def sigmoid(self):
        result = Tensor(1 / (1 + mx.exp(-self.data)), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'sigmoid'
        return result
    
    def tanh(self):
        result = Tensor(mx.tanh(self.data), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'tanh'
        return result
    
    def softmax(self):
        shifted = self - self.max()     # Tensor
        exps = shifted.exp()           # Tensor
        summed = exps.sum()            # Scalar Tensor
        result = exps / summed         # Tensor
        result._extra = 'softmax'

        return result
    
    def exp(self):
        result = Tensor(mx.exp(self.data), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'exp'
        return result
    
    def log(self, base: float = 10.0):
        if base == mx.e:
            data = mx.log(self.data)
        else:
            data = mx.log(self.data) / mx.log(mx.array(base))

        result = Tensor(data, requires_grad=self.requires_grad)
        result._parents = [self, Tensor(base)]
        result._op = "log"
        return result
    
    def step(self):
        out = Tensor((self.data > 0).astype(self.data.dtype), requires_grad=self.requires_grad)
        out._parents = [self]
        out._op = 'step'
    
    def reshape(self, *shape):
        result = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'reshape'
        return result
    
    def T(self):
        result = Tensor(self.data.T, requires_grad=self.requires_grad)
        result._parents = [self]
        result._op = 'transpose'
        return result
    
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = mx.ones_like(self.data)  # Start with dL/dself = 1 for scalars
        
        if self.grad is None:
            self.grad = mx.array(grad)
        else:
            self.grad += grad  # Accumulate gradients if needed
        
        for parent in self._parents:
            if parent.requires_grad:
                if self._op == 'add':
                    parent.backward(grad)  # d(a+b)/da = 1, d(a+b)/db = 1
                elif self._op == 'sub':
                    parent.backward(grad if parent is self._parents[0] else -grad)  # d(a-b)/da = 1, d(a-b)/db = -1
                elif self._op == 'neg':
                    parent.backward(-grad)
                elif self._op == 'mul':
                    parent.backward(grad * (self._parents[1].data if parent is self._parents[0] else self._parents[0].data))  # d(a*b)/da = b, d(a*b)/db = a
                elif self._op == 'div':
                    parent.backward((grad/self._parents[1].data if parent is self._parents[0] else (-self._parents[0].data @ grad) / (self._parents[1].data**2)))  # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                elif self._op == 'pow':
                    parent.backward(grad * self._parents[1].data * (parent.data ** (self._parents[1].data - 1)))  # d(a^b)/da = b * a^(b-1)
                elif self._op == 'matmul':
                    parent.backward(mx.matmul(grad, self._parents[1].data.T) if parent is self._parents[0] else mx.matmul(self._parents[0].data.T, grad))  # d(A @ B)/dA = dL/dC @ B.T, d(A @ B)/dB = A.T @ dL/dC
                elif self._op == 'sum':
                    parent.backward(mx.ones_like(parent.data) * grad)
                elif self._op == 'mean':
                    parent.backward(mx.ones_like(parent.data) * grad / parent.data.size)
                elif self._op == 'relu':
                    parent.backward(grad * (self.data > 0))
                elif self._op == 'sigmoid':
                    sigmoid_val = 1 / (1 + mx.exp(-parent.data))
                    parent.backward(grad * sigmoid_val * (1 - sigmoid_val))
                elif self._op == 'tanh':
                    parent.backward(grad * (1 - mx.tanh(parent.data)**2))
                elif self._op == 'step':
                    parent.backward(mx.zeros_like(self.data))
                elif self._op == 'exp':
                    parent.backward(grad * self.data)
                elif self._op == 'log':
                    base = self._parents[1]
                    if base == mx.e:
                        parent.backward(self.grad / parent.data)
                    else:
                        parent.backward(self.grad / (parent.data * mx.log(mx.array(base.data))))
                elif self._op == 'reshape':
                    parent.backward(grad.reshape(parent.data.shape))
                elif self._op == 'transpose':
                    parent.backward(grad.T)
                elif self._op == 'getitem':
                    grad_full = mx.zeros_like(parent.data)
                    grad_full[self._idx] = grad
                    parent.backward(grad_full)

        if self._op == 'apply':
            for parent in self._parents:
                parent.backward()
