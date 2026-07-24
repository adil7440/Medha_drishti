import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.plotly_charts import (
    create_histogram_comparison,
    create_radar_chart,
    create_metrics_boxplot,
    create_scatter_plot
)

st.header("Quantitative Analytics and Graphs")

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
metrics_path = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"

if not metrics_path.exists():
    st.warning("Please run stage2_pipeline.py first.")
    st.stop()

df = pd.read_csv(metrics_path)

st.subheader("1. Multi-Metric Quality Radar Profile")
mean_dict = df.mean(numeric_only=True).to_dict()
fig_radar = create_radar_chart(mean_dict)
st.plotly_chart(fig_radar, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.subheader("2. Quality Metric Distribution Boxplots")
    selected_metric = st.selectbox("Select Metric for Boxplot", options=["PSNR", "SSIM", "RMSE", "UQI", "FSIM"])
    fig_box = create_metrics_boxplot(df, metric_col=selected_metric)
    st.plotly_chart(fig_box, use_container_width=True)

with c2:
    st.subheader("3. Scatter Comparison Plot")
    fig_scatter = create_scatter_plot(df, x_col="PSNR", y_col="SSIM")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("4. Intensity Distribution Histogram (Before vs After)")
npz_files = list(preprocessed_dir.glob("*.npz"))
if npz_files:
    data = np.load(npz_files[0])
    fig_hist = create_histogram_comparison(data["orig_slice"], data["stage_final"])
    st.plotly_chart(fig_hist, use_container_width=True)
