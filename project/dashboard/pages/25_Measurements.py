import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="25 - Clinical Measurements", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("📏 25 - Clinical Measurements")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

meas = result['measurements']
col1, col2, col3 = st.columns(3)
with col1: st.markdown(f'<div class="metric-card"><div>Volume</div><div class="metric-value">{meas["Volume_mm3"]} mm³</div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><div>Max Area</div><div class="metric-value">{meas["Area_mm2_max_slice"]} mm²</div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><div>Affected Tissue %</div><div class="metric-value">{meas["Affected_Tissue_Percentage"]}%</div></div>', unsafe_allow_html=True)
st.json(meas)
