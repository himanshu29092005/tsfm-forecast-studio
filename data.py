import numpy as np, pandas as pd

def synthetic_load(n=24 * 60, seed=0):
    """Hourly 'electricity load' style series: daily + weekly seasonality, trend, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (100 + 0.02 * t + 15 * np.sin(2 * np.pi * t / 24)
         + 8 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 2.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"date": idx, "value": y})

def load_csv(file, value_col=None):
    df = pd.read_csv(file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    value_col = value_col or df.select_dtypes("number").columns[-1]
    return df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
