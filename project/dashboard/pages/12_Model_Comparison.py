import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

st.markdown("""
<div class="main-header">
    <h1 style="color:#0ea5e9; margin:0 0 4px 0;">Model Comparison</h1>
    <p>Head-to-head comparison of all trained MRI enhancement models across all metrics.</p>
</div>
""", unsafe_allow_html=True)

metrics_csv = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
if not metrics_csv.exists():
    st.warning("No evaluation data found. Run `run_stage3.py` first.")
    st.stop()

df = pd.read_csv(metrics_csv)

# Overview
st.markdown("### Model Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Models Trained", len(df))
with c2:
    st.metric("Best PSNR", f"{df['PSNR'].max():.2f} dB")
with c3:
    st.metric("Best SSIM", f"{df['SSIM'].max():.4f}")
with c4:
    best = df.iloc[0]["Model"]
    st.metric("Top Model", best)

st.markdown("---")

# PSNR & SSIM Bars
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(df, x="Model", y="PSNR", color="Model", title="PSNR (dB) Comparison",
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(df, x="Model", y="SSIM", color="Model", title="SSIM Index Comparison",
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# LPIPS, NIQE, BRISQUE (lower is better)
c1, c2, c3 = st.columns(3)
with c1:
    fig = px.bar(df, x="Model", y="LPIPS", color="Model", title="LPIPS (lower=better)",
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(df, x="Model", y="NIQE", color="Model", title="NIQE (lower=better)",
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    fig = px.bar(df, x="Model", y="BRISQUE", color="Model", title="BRISQUE (lower=better)",
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Radar chart
st.markdown("### Performance Radar Chart")
radar_metrics = ["PSNR", "SSIM", "FSIM", "VIF", "UQI"]
fig = go.Figure()
for i, row in df.iterrows():
    values = [row[m] / df[m].max() for m in radar_metrics]
    values.append(values[0])
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_metrics + [radar_metrics[0]],
        name=row["Model"],
        fill='toself',
        opacity=0.6,
    ))
fig.update_layout(template="plotly_dark", title="Normalized Model Performance", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
st.plotly_chart(fig, use_container_width=True)

# Inference & Size
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(df, x="Model", y="Inference_Time_ms", color="Model",
                 title="Inference Time (ms)", template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(df, x="Model", y="Model_Size_MB", color="Model",
                 title="Model Size (MB)", template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Full table
st.markdown("### Complete Evaluation Table")
display_cols = [c for c in df.columns if c not in ["Composite_Score"]]
st.dataframe(df[display_cols], use_container_width=True)
