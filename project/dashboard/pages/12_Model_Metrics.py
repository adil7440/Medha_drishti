import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.metrics_cards import render_metric_card

st.header("AI Model Performance Metrics")

metrics_csv = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
if not metrics_csv.exists():
    st.warning("No Stage 3 evaluation metrics found. Please run `run_stage3.py` first.")
    st.stop()

df = pd.read_csv(metrics_csv)

model_name = st.radio("Select Model Metrics View", options=df["Model"].unique())
model_row = df[df["Model"] == model_name].iloc[0]

st.markdown(f"### Benchmark Metrics: {model_name}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("PSNR", f"{model_row['PSNR']:.2f} dB", is_positive=True)
    render_metric_card("MSE", f"{model_row['MSE']:.6f}")
with c2:
    render_metric_card("SSIM", f"{model_row['SSIM']:.4f}", is_positive=True)
    render_metric_card("RMSE", f"{model_row['RMSE']:.4f}")
with c3:
    render_metric_card("LPIPS", f"{model_row['LPIPS']:.4f}", is_positive=True)
    render_metric_card("Inference Speed", f"{model_row['Inference_Time_ms']} ms/slice")
with c4:
    render_metric_card("Model Size", f"{model_row['Model_Size_MB']} MB")
    render_metric_card("Parameters", f"{model_row['Params']:,}")

st.markdown("### Full Evaluation Metrics Table")
st.dataframe(df, use_container_width=True)
