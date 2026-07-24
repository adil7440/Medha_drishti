import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.metrics_cards import render_summary_row, render_metric_card

st.header("Quantitative Quality Metrics Summary")

metrics_path = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"
if not metrics_path.exists():
    st.warning("Please run the Stage 2 pipeline first to populate metrics.")
    st.stop()

df = pd.read_csv(metrics_path)

st.markdown("### Average Preprocessing Improvements Across All Volumes")
mean_dict = df.mean(numeric_only=True).to_dict()
render_summary_row(mean_dict)

st.markdown("---")
st.markdown("### Complete 17 Quality Metrics Breakdown")

c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("MSE", f"{mean_dict.get('MSE', 0):.6f}")
    render_metric_card("RMSE", f"{mean_dict.get('RMSE', 0):.4f}")
    render_metric_card("UQI", f"{mean_dict.get('UQI', 0):.4f}")
    render_metric_card("FSIM", f"{mean_dict.get('FSIM', 0):.4f}")
    render_metric_card("GMSD", f"{mean_dict.get('GMSD', 0):.4f}")
    render_metric_card("VIF", f"{mean_dict.get('VIF', 0):.4f}")

with c2:
    render_metric_card("BRISQUE", f"{mean_dict.get('BRISQUE', 0):.2f}")
    render_metric_card("NIQE", f"{mean_dict.get('NIQE', 0):.2f}")
    render_metric_card("PIQE", f"{mean_dict.get('PIQE', 0):.2f}")
    render_metric_card("LPIPS (Proxy)", f"{mean_dict.get('LPIPS', 0):.4f}")
    render_metric_card("Sharpness (Before)", f"{mean_dict.get('Sharpness_Before', 0):.4f}")
    render_metric_card("Sharpness (After)", f"{mean_dict.get('Sharpness_After', 0):.4f}")

with c3:
    render_metric_card("Edge Strength (Before)", f"{mean_dict.get('EdgeStrength_Before', 0):.4f}")
    render_metric_card("Edge Strength (After)", f"{mean_dict.get('EdgeStrength_After', 0):.4f}")
    render_metric_card("Noise Level (Before)", f"{mean_dict.get('NoiseLevel_Before', 0):.4f}")
    render_metric_card("Noise Level (After)", f"{mean_dict.get('NoiseLevel_After', 0):.4f}")

st.markdown("### Processed Metrics Table")
st.dataframe(df, use_container_width=True)
