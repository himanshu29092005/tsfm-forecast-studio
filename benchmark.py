"""CLI: python benchmark.py [--csv file.csv --col OT] [--h 48] [--m 24]"""
import argparse, numpy as np, pandas as pd
from data import synthetic_load, load_csv
from models import MODELS
from metrics import mae, rmse, mase

def run(y, h, m, names):
    train, test = y[:-h], y[-h:]
    res, fc = {}, {}
    for n in names:
        try:
            f = np.asarray(MODELS[n](train, h, m))
            fc[n] = f
            res[n] = {"MAE": mae(test, f), "RMSE": rmse(test, f), "MASE": mase(test, f, train, m)}
        except Exception as e:
            print(f"[skip] {n}: {type(e).__name__}: {str(e)[:80]}")
    return train, test, res, fc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--col")
    ap.add_argument("--h", type=int, default=48); ap.add_argument("--m", type=int, default=24)
    a = ap.parse_args()
    df = load_csv(a.csv, a.col) if a.csv else synthetic_load()
    _, _, res, _ = run(df["value"].values, a.h, a.m, list(MODELS))
    out = pd.DataFrame(res).T.sort_values("MASE")
    print(out.round(3)); out.round(4).to_csv("results.csv")
