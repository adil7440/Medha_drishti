import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="22 - Enhancement Viewer", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("✨ 22 - Enhancement Viewer")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

vol_pre = st.session_state['vol_preprocessed']
vol_enh = st.session_state['vol_enhanced']
z = st.slider("Z-Slice", 0, vol_enh.shape[0]-1, vol_enh.shape[0]//2)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Stage 2 Output")
    fig = go.Figure(data=go.Heatmap(z=vol_pre[z], colorscale='gray', showscale=False))
    fig.update_layout(width=400, height=400, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Stage 3 U-DnCNN Enhanced")
    fig2 = go.Figure(data=go.Heatmap(z=vol_enh[z], colorscale='gray', showscale=False))
    fig2.update_layout(width=400, height=400, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)
