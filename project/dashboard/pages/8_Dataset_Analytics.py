import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.header("Dataset-Wide Preprocessing Analytics")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
metrics_path = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"

if not metrics_path.exists():
    st.warning("Please run stage2_pipeline.py first.")
    st.stop()

df = pd.read_csv(metrics_path)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Patient & Modality Distribution")
    fig_pie = px.pie(df, names="Dataset", title="Volume Count by Dataset", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.subheader("Processing Time Benchmark (Sec)")
    fig_time = px.histogram(df, x="Processing_Time_Sec", color="Dataset", title="Processing Time Distribution per Volume", template="plotly_dark")
    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("### Summary Statistics Table")
st.dataframe(df.groupby(["Dataset", "Modality"]).mean(numeric_only=True).reset_index(), use_container_width=True)
