from tensor import Tensor
import mlx.core as mx

def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    data = labels.data.astype(mx.int32)  # or mx.int64 depending on your platform/setup
    out = mx.zeros((data.size, num_classes))
    out[mx.arange(data.size), data] = 1.0
    return Tensor(out, requires_grad=False)


class Loss:
    def __call__(self, prediction: Tensor, target: Tensor):
        return self.forward(prediction, target)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, prediction, target):
        diff = prediction - target
        return (diff * diff).mean()


class CrossEntropyLoss(Loss):
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction._extra != 'softmax':
            raise UserWarning('CrossEtropyLoss was used on a model that does not output softmax values, is this the intended behavior')
        # Convert targets to one-hot
        num_classes = prediction.shape[0]
        one_hot_targets = one_hot(target, num_classes).data  # raw data for smoothing

        if self.label_smoothing > 0:
            smooth = self.label_smoothing
            one_hot_targets = (1 - smooth) * one_hot_targets + smooth / num_classes

        # Convert back to Tensor for log and loss operations
        target_tensor = Tensor(one_hot_targets, requires_grad=False)

        log_preds = prediction.log(base=mx.e)
        loss = - (target_tensor * log_preds).sum() / prediction.data.shape[0]
        return loss
    
if __name__ == '__main__':
    loss = CrossEntropyLoss()
    loss(Tensor([0.1,0.64,0.14,0.8,0.4]), Tensor([1]))