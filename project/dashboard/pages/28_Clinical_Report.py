import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="28 - Clinical Report", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("📄 28 - Clinical Report")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.report_generator_stage4 import Stage4ReportGenerator
st.markdown("### Automatically Generated Radiology Report")
st.info("The system is ready to compile the patient data, 3D measurements, and diagnostic heatmaps into a secure PDF.")
if st.button("Generate Final PDF"):
    with st.spinner("Compiling PDF..."):
        Stage4ReportGenerator.generate_report(result, {}, "project/stage4/reports/final_report.pdf")
    st.success("PDF generated successfully! Head over to the Downloads tab.")
