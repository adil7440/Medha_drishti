import scipy.ndimage as ndimage
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="27 - Explainability Heatmap", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("🔥 27 - Explainability Heatmap")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

vol = st.session_state['vol_enhanced']
mask = st.session_state['mask']
z = st.slider("Z-Slice", 0, vol.shape[0]-1, result['measurements']['Most_Affected_Slice'])
st.write("Simulated Grad-CAM Attention Map highlighting the detected structural anomalies.")
# Simulate heatmap by blurring the mask and mapping it to Jet
heatmap = ndimage.gaussian_filter(mask[z].astype(float), sigma=3)
fig = go.Figure(data=go.Heatmap(z=vol[z], colorscale='gray', showscale=False))
if np.any(heatmap):
    fig.add_trace(go.Heatmap(z=np.where(heatmap>0.1, heatmap, None), colorscale='Jet', opacity=0.4, showscale=False))
fig.update_layout(width=600, height=600, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)
