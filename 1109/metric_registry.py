import numpy as np

class MetricRegistry:
    def __init__(self):
        self._handlers = {}

    def register(self, key: str, title: str, fn):
        self._handlers[key] = {"title": title, "fn": fn}

    def compute(self, key: str, a: np.ndarray, b: np.ndarray) -> float:
        return self._handlers[key]["fn"](a, b)

    def title(self, key: str) -> str:
        return self._handlers[key]["title"]

    def keys(self):
        return list(self._handlers.keys())


Metrics = MetricRegistry()

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("無有效資料可比較")
    a = a[mask]; b = b[mask]
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))

Metrics.register("mse", "Mean Squared Error", _mse)

# 你可以額外加：
# Metrics.register("mae", "Mean Absolute Error", lambda a,b: float(np.mean(np.abs(a-b))))
# Metrics.register("rmse", "Root MSE", lambda a,b: float(np.sqrt(np.mean((a-b)**2))))
