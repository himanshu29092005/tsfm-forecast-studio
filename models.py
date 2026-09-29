import warnings
import numpy as np
warnings.filterwarnings("ignore")

def naive(train, h, m=1):
    return np.repeat(train[-1], h)

def seasonal_naive(train, h, m=24):
    reps = int(np.ceil(h / m))
    return np.tile(train[-m:], reps)[:h]

def holt_winters(train, h, m=24):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=m).fit()
    return np.asarray(fit.forecast(h))

def arima(train, h, m=24):
    from statsmodels.tsa.arima.model import ARIMA
    fit = ARIMA(train[-600:], order=(2, 1, 2)).fit()   # last 600 pts keeps it fast
    return np.asarray(fit.forecast(h))

def mlp_lag(train, h, m=24, lags=48):
    """Small neural net on lagged windows, recursive multi-step forecast."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    mu, sd = train.mean(), train.std() + 1e-9
    z = (train - mu) / sd
    X = np.array([z[i:i + lags] for i in range(len(z) - lags)])
    y = z[lags:]
    net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0).fit(X, y)
    win, out = list(z[-lags:]), []
    for _ in range(h):
        p = net.predict(np.array(win[-lags:])[None])[0]
        out.append(p); win.append(p)
    return np.array(out) * sd + mu

def chronos_zero_shot(train, h, m=24, name="amazon/chronos-t5-small"):
    """Zero-shot pretrained time-series foundation model (needs torch + chronos-forecasting)."""
    import torch
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(train[-512:], dtype=torch.float32)
    samples = pipe.predict(ctx, h)                       # [1, n_samples, h]
    return np.median(samples[0].numpy(), axis=0)

MODELS = {"Naive": naive, "Seasonal Naive": seasonal_naive, "Holt-Winters": holt_winters,
          "ARIMA": arima, "MLP (lags)": mlp_lag, "Chronos (zero-shot)": chronos_zero_shot}
