import numpy as np

def mae(y, f): return float(np.mean(np.abs(y - f)))
def rmse(y, f): return float(np.sqrt(np.mean((y - f) ** 2)))

def mase(y, f, train, m=1):
    """Scaled by in-sample seasonal-naive error (Hyndman & Koehler, 2006)."""
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    return float(np.mean(np.abs(y - f)) / (scale + 1e-12))
