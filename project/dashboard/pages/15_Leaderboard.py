import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

st.markdown("""
<div class="main-header">
    <h1 style="color:#10b981; margin:0 0 4px 0;">Leaderboard</h1>
    <p>Automatic model ranking using weighted composite scoring across all quality metrics.</p>
</div>
""", unsafe_allow_html=True)

metrics_csv = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
if not metrics_csv.exists():
    st.warning("No leaderboard data found. Run `run_stage3.py` first.")
    st.stop()

df = pd.read_csv(metrics_csv)
best = df.iloc[0]

# Winner Banner
st.markdown(f"""
<div style="background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981;
            border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;">
    <h2 style="color: #10b981; margin: 0 0 8px 0;">Champion Model: {best['Model']}</h2>
    <p style="color: #a7f3d0; margin: 0; font-size: 1.1rem;">
        Rank #1 | Composite Score: <strong>{best['Composite_Score']:.4f}</strong> |
        PSNR: <strong>{best['PSNR']:.4f} dB</strong> |
        SSIM: <strong>{best['SSIM']:.4f}</strong> |
        Inference: <strong>{best['Inference_Time_ms']:.1f} ms</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Ranking Table
st.markdown("### Official Leaderboard")
display_cols = ["Rank", "Model", "PSNR", "SSIM", "LPIPS", "NIQE", "BRISQUE", "Inference_Time_ms", "Model_Size_MB", "Composite_Score"]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

# Ranking Methodology
st.markdown("---")
st.markdown("### Ranking Methodology")
st.markdown("""
| Metric | Direction | Weight |
|--------|-----------|--------|
| PSNR | Higher is better | 0.30 |
| SSIM | Higher is better | 0.25 |
| LPIPS | Lower is better | 0.15 |
| NIQE | Lower is better | 0.15 |
| BRISQUE | Lower is better | 0.15 |

Each metric is min-max normalized across all models to [0, 1] before weighting.
Lower-is-better metrics are inverted (1 - normalized) so higher scores always mean better performance.
""")

# Visualizations
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(df, x="Model", y="Composite_Score", color="Model",
                 title="Composite Score Ranking", template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    radar_m = ["PSNR", "SSIM", "FSIM"]
    for i, row in df.iterrows():
        vals = [row[m] / df[m].max() for m in radar_m]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=radar_m + [radar_m[0]],
            name=row["Model"], fill='toself', opacity=0.5,
        ))
    fig.update_layout(template="plotly_dark", title="Performance Radar", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

# Trophy visualization
st.markdown("### Model Podium")
medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
for i, row in df.head(3).iterrows():
    medal = ["Gold", "Silver", "Bronze"][i]
    color = medal_colors[i]
    st.markdown(f"""
    <div style="background:{color}22; border: 2px solid {color}; border-radius: 8px;
                padding: 16px 24px; margin-bottom: 8px;">
        <span style="font-size: 1.5rem; font-weight: 800; color: {color};">#{row['Rank']} {medal}</span>
        <span style="font-size: 1.2rem; color: #f9fafb; margin-left: 16px;">{row['Model']}</span>
        <span style="float: right; color: #9ca3af;">Score: {row['Composite_Score']:.4f} | PSNR: {row['PSNR']:.4f} dB</span>
    </div>
    """, unsafe_allow_html=True)
