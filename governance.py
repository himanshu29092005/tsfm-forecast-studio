import numpy as np, datetime as dt

def psi(ref, cur, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate drift, >0.25 major drift."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

def drift_label(v):
    return "stable" if v < 0.1 else "moderate drift" if v < 0.25 else "major drift"

def anomalies(y, window=48, z=3.5):
    """Rolling median/MAD anomaly flags."""
    import pandas as pd
    s = pd.Series(y)
    med = s.rolling(window, min_periods=10).median()
    mad = (s - med).abs().rolling(window, min_periods=10).median() + 1e-9
    return ((s - med).abs() / (1.4826 * mad) > z).values

def model_card(results, horizon, season, n_train, drift):
    best = min(results, key=lambda k: results[k]["MASE"])
    rows = "\n".join(f"| {k} | {v['MAE']:.3f} | {v['RMSE']:.3f} | {v['MASE']:.3f} |" for k, v in results.items())
    return f"""# Model Card - Time-Series Forecasting Benchmark
Generated: {dt.datetime.now():%Y-%m-%d %H:%M}

**Intended use:** short-term univariate forecasting (research / decision support, not autonomous control).
**Data:** {n_train} training points, seasonality m={season}, horizon={horizon}.
**Best model (by MASE):** {best}

| Model | MAE | RMSE | MASE |
|---|---|---|---|
{rows}

**Drift check (PSI, train vs. recent window):** {drift:.3f} ({drift_label(drift)})
**Limitations:** univariate only; no exogenous variables; foundation model used zero-shot without fine-tuning.
**Risks:** performance degrades under regime change; monitor drift and re-benchmark periodically.
"""
