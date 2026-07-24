import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="20 - MRI Metadata", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("📊 20 - MRI Metadata")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card"><div>Patient ID</div><div class="metric-value">{result["patient_id"]}</div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><div>Modality</div><div class="metric-value">{result["modality"]}</div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><div>Volume Shape</div><div class="metric-value">{result["shape"]}</div></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><div>Voxel Spacing (mm)</div><div class="metric-value">{"x".join([f"{x:.2f}" for x in result["voxel_spacing"]])}</div></div>', unsafe_allow_html=True)
st.json(result)
