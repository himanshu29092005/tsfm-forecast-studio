# TSFM Forecast Studio
Benchmark of a **pretrained time-series foundation model (Amazon Chronos, zero-shot)** against Naive, Seasonal-Naive,
Holt-Winters, ARIMA and a lag-based neural net (MLP), wrapped in a **Streamlit** app with anomaly detection and
AI-governance tooling (PSI drift monitoring + auto-generated model card).

## Run
```bash
pip install -r requirements.txt
python benchmark.py --h 48 --m 24          # CLI benchmark -> results.csv
streamlit run app.py                       # interactive app
# optional foundation model: pip install torch chronos-forecasting
```
Use real data: download ETTh1 (https://github.com/zhouhaoyi/ETDataset), upload `ETTh1.csv`, pick column `OT`.

## Metrics
MAE, RMSE, MASE (scaled by seasonal-naive in-sample error).

## Results
Paste your `results.csv` table here after running on ETTh1.

## References
Ansari et al., *Chronos: Learning the Language of Time Series* (2024); Das et al., *TimesFM* (2024); Hyndman & Koehler (2006).
