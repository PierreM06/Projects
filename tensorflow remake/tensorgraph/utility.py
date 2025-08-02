import time
from .tensor import Tensor
import mlx.core as mx

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min", monitor: str = ""):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = float('inf') if mode == "min" else -float('inf')
        self.wait = 0
        self.stopped_epoch = None
        self.should_stop = False

    def check(self, current: float, epoch: int) -> bool:
        if self._is_improvement(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
        return self.should_stop

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta


class ProgressBar:
    def __init__(self, max: int, length: int = 20, title: str = '') -> None:
        self.current = 0
        self.max = max
        self.length = length
        self.title = title
        self.start = time.time()
        self.avg = 0.0
        self.avg_weight = 0
        self.last_call = self.start
        self.max_digits = len(str(self.max))

    def __call__(self) -> None:
        now = time.time()
        step_time = now - self.last_call
        self.last_call = now

        # Update weighted average
        self.avg = (self.avg * self.avg_weight + step_time) / (self.avg_weight + 1)
        self.avg_weight += 1

        progress_ratio = (self.current+1) / self.max
        filled = int(progress_ratio * self.length)
        bar = '#' * filled + ('>' if progress_ratio != 1 else '') + '-' * (self.length - filled - 1)

        elapsed_sec = now - self.start
        remaining_sec = self.avg * (self.max - self.current)
        time_per_item_ms = int(self.avg * 1000)

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))

        prefix = f"{self.title} | " if self.title else ""
        progress_str = f'{self.current+1:>{self.max_digits}d}/{self.max}'


        print(
            f'{prefix}{progress_str} | {bar} | '
            f'Time/item: {time_per_item_ms:3d}ms | '
            f'ETA: {eta_str} | '
            f'Elapsed: {elapsed_str}',
            end='\r',
            flush=True
        )

        self.current += 1

        if self.current == self.max:
            print()

# def bincount(t: mx.array, num_classes: int):
#     result = mx.zeros((num_classes,), dtype=mx.int32)
#     for i in range(num_classes):
#         result[i] = mx.sum(t == i)
#     return result

def bincount(t: mx.array, num_classes: int) -> mx.array:
    ones = mx.ones_like(t)
    counts = mx.zeros((num_classes,), dtype=mx.int32)
    
    # Count all values (including -1, which wraps to last index)
    counts = counts.at[t.astype(mx.int32)].add(ones)
    
    # Subtract how many -1s there were
    num_neg_ones = mx.sum(t == -1).astype(mx.int32)
    counts = counts.at[-1].add(-num_neg_ones)  # Remove erroneous additions to the last bin
    
    return counts
