import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="24 - Disease Detection", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("🏥 24 - Disease Detection")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

color = "#10b981" if result["disease_class"] == "Normal" else "#ef4444"
html_block = f"""
<div style="background: rgba({239 if color=='#ef4444' else 16}, {68 if color=='#ef4444' else 185}, {68 if color=='#ef4444' else 129}, 0.1); 
            border: 2px solid {color}; border-radius: 12px; padding: 30px; text-align: center;">
    <h3 style="color: #9ca3af; margin-bottom: 5px; font-weight: normal;">Disease Class:</h3>
    <h1 style="color: {color}; margin-top: 0; font-size: 42px;">{result['disease_class']}</h1>
    <hr style="border-color: {color}; opacity: 0.3; margin: 20px 0;">
    <h4 style="color: #9ca3af; margin-bottom: 5px; font-weight: normal;">Diagnosis:</h4>
    <h2 style="color: {color}; margin-top: 0;">{result['diagnosis']}</h2>
</div>
"""
st.markdown(html_block, unsafe_allow_html=True)
