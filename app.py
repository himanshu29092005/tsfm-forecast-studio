import numpy as np, pandas as pd, streamlit as st
from data import synthetic_load, load_csv
from models import MODELS
from benchmark import run
from governance import psi, drift_label, anomalies, model_card

st.set_page_config(page_title="TSFM Forecast Studio", layout="wide")
st.title("Time-Series Foundation Model Forecast Studio")
st.caption("Zero-shot foundation model (Chronos) vs classical & neural baselines, with governance checks.")

with st.sidebar:
    up = st.file_uploader("Upload CSV (first col = date)", type="csv")
    df = load_csv(up) if up else synthetic_load()
    if up:
        col = st.selectbox("Value column", pd.read_csv(up, nrows=1).select_dtypes("number").columns)
        up.seek(0); df = load_csv(up, col)
    h = st.slider("Horizon", 12, 168, 48)
    m = st.number_input("Season length", 1, 200, 24)
    names = st.multiselect("Models", list(MODELS), default=["Naive", "Seasonal Naive", "Holt-Winters", "MLP (lags)"])
    go = st.button("Run benchmark", type="primary")

y = df["value"].values
t1, t2, t3 = st.tabs(["Forecast", "Anomalies", "Governance"])
if go:
    with st.spinner("Running models..."):
        train, test, res, fc = run(y, h, m, names)
    st.session_state.update(res=res, fc=fc, train=train, test=test)

if "res" in st.session_state:
    res, fc, train, test = (st.session_state[k] for k in ("res", "fc", "train", "test"))
    with t1:
        plot = pd.DataFrame({"Actual": np.concatenate([train[-3 * h:], test])},
                            index=range(-3 * h, h))
        for n, f in fc.items():
            plot[n] = np.nan; plot.loc[0:h - 1, n] = f
        st.line_chart(plot)
        st.dataframe(pd.DataFrame(res).T.sort_values("MASE").style.format("{:.3f}"))
    with t3:
        d = psi(train[:len(train) // 2], train[-len(train) // 4:])
        st.metric("Drift PSI (early vs recent)", f"{d:.3f}", drift_label(d))
        card = model_card(res, h, m, len(train), d)
        st.markdown(card); st.download_button("Download model card", card, "MODEL_CARD.md")
else:
    with t1: st.info("Choose models in the sidebar and click **Run benchmark**.")
with t2:
    flags = anomalies(y)
    st.write(f"{int(flags.sum())} anomalies flagged (rolling median/MAD)")
    a = pd.DataFrame({"value": y, "anomaly": np.where(flags, y, np.nan)}, index=df["date"])
    st.line_chart(a)
