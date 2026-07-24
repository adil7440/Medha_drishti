import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.header("Stage 3 Leaderboard and Model Comparison")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
metrics_csv = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"

if not metrics_csv.exists():
    st.warning("No Stage 3 model evaluation data found. Run `run_stage3.py` to generate leaderboard.")
    st.stop()

df = pd.read_csv(metrics_csv)
best_model = df.iloc[0]

st.markdown(
    f"""
    <div class="main-header">
        <h2 style="color:#10b981; margin:0 0 4px 0;">Top Model Selected: {best_model['Model']}</h2>
        <p>Rank 1 | Peak PSNR: {best_model['PSNR']} dB | Peak SSIM: {best_model['SSIM']} | Inference: {best_model['Inference_Time_ms']} ms/slice</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Official Model Leaderboard")
st.dataframe(
    df[["Rank", "Model", "PSNR", "SSIM", "LPIPS", "Inference_Time_ms", "Model_Size_MB", "Params"]],
    use_container_width=True
)

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("PSNR Comparison")
    fig_psnr = px.bar(df, x="Model", y="PSNR", color="Model", title="PSNR (dB) by Model", template="plotly_dark", color_discrete_sequence=["#0ea5e9", "#10b981"])
    st.plotly_chart(fig_psnr, use_container_width=True)

with c2:
    st.subheader("SSIM Comparison")
    fig_ssim = px.bar(df, x="Model", y="SSIM", color="Model", title="SSIM Index by Model", template="plotly_dark", color_discrete_sequence=["#0ea5e9", "#10b981"])
    st.plotly_chart(fig_ssim, use_container_width=True)
