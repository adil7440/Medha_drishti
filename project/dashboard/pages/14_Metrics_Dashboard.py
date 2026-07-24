import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.metrics_cards import render_metric_card

st.markdown("""
<div class="main-header">
    <h1 style="color:#8b5cf6; margin:0 0 4px 0;">Metrics Dashboard</h1>
    <p>Comprehensive evaluation metrics for all trained MRI enhancement models.</p>
</div>
""", unsafe_allow_html=True)

metrics_csv = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
if not metrics_csv.exists():
    st.warning("No evaluation metrics found. Run `run_stage3.py` first.")
    st.stop()

df = pd.read_csv(metrics_csv)

model_name = st.radio("Select Model", df["Model"].unique(), horizontal=True)
model_row = df[df["Model"] == model_name].iloc[0]

st.markdown(f"### Metrics: {model_name}")

# Row 1: Core metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("PSNR", f"{model_row['PSNR']:.4f} dB", is_positive=True)
    render_metric_card("SSIM", f"{model_row['SSIM']:.4f}", is_positive=True)
with c2:
    render_metric_card("LPIPS", f"{model_row['LPIPS']:.4f}", is_positive=False)
    render_metric_card("FSIM", f"{model_row['FSIM']:.4f}", is_positive=True)
with c3:
    render_metric_card("VIF", f"{model_row['VIF']:.4f}", is_positive=True)
    render_metric_card("UQI", f"{model_row['UQI']:.4f}", is_positive=True)
with c4:
    render_metric_card("MSE", f"{model_row['MSE']:.6f}")
    render_metric_card("RMSE", f"{model_row['RMSE']:.4f}")

# Row 2: No-reference metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("NIQE", f"{model_row['NIQE']:.4f}", is_positive=False)
    render_metric_card("BRISQUE", f"{model_row['BRISQUE']:.4f}", is_positive=False)
with c2:
    render_metric_card("PIQE", f"{model_row['PIQE']:.4f}", is_positive=False)
    render_metric_card("Entropy", f"{model_row['Entropy']:.4f}", is_positive=True)
with c3:
    render_metric_card("Contrast", f"{model_row['Contrast']:.4f}", is_positive=True)
    render_metric_card("Sharpness", f"{model_row['Sharpness']:.4f}", is_positive=True)
with c4:
    render_metric_card("Edge Strength", f"{model_row['EdgeStrength']:.4f}", is_positive=True)
    render_metric_card("Noise Level", f"{model_row['NoiseLevel']:.4f}", is_positive=False)

# Row 3: Infrastructure
st.markdown("### Infrastructure Metrics")
c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("Inference Time", f"{model_row['Inference_Time_ms']:.1f} ms/slice")
with c2:
    render_metric_card("GPU Memory", f"{model_row['GPU_Memory_MB']:.1f} MB")
with c3:
    render_metric_card("Model Size", f"{model_row['Model_Size_MB']:.2f} MB | {int(model_row['Params']):,} params")

# Cross-model comparison charts
st.markdown("---")
st.markdown("### Cross-Model Metric Comparison")

metric_groups = {
    "Quality (Higher=Better)": ["PSNR", "SSIM", "FSIM", "VIF", "UQI"],
    "Quality (Lower=Better)": ["LPIPS", "NIQE", "BRISQUE", "PIQE", "RMSE"],
    "Statistical": ["Entropy", "Contrast", "Sharpness", "EdgeStrength", "NoiseLevel"],
}

for group_name, metrics in metric_groups.items():
    st.markdown(f"#### {group_name}")
    available = [m for m in metrics if m in df.columns]
    if available:
        fig = go.Figure()
        for metric in available:
            vals = df[metric].values
            v_max = vals.max()
            norm = vals / v_max if v_max > 0 else vals
            fig.add_trace(go.Bar(
                name=metric,
                x=df["Model"],
                y=norm,
                text=[f"{v:.4f}" for v in vals],
                textposition='outside',
            ))
        fig.update_layout(barmode='group', template="plotly_dark", title=group_name)
        st.plotly_chart(fig, use_container_width=True)

# Full table
st.markdown("### Complete Metrics Table")
st.dataframe(df, use_container_width=True)
