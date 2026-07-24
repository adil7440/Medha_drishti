import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DARK_BG = '#0b0f19'
st.set_page_config(page_title="29 - Download Center", layout="wide", page_icon="🧠")
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; color: #f9fafb; }}
    .metric-card {{ background: #111827; border-radius: 12px; padding: 20px; border: 1px solid #1f2937; text-align: center; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: #0ea5e9; }}
    </style>
""", unsafe_allow_html=True)
st.title("💾 29 - Download Center")

if 'stage4_result' not in st.session_state:
    st.warning("Please upload and analyze an MRI volume in '19_Upload_MRI.py' first.")
    st.stop()

result = st.session_state['stage4_result']

st.markdown("### Medical Exports")
pdf_path = Path("project/stage4/reports/final_report.pdf")
if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        st.download_button("Download Clinical PDF Report", f.read(), "Clinical_Report.pdf", "application/pdf", type="primary")
else:
    st.warning("No PDF generated yet. Please generate it in Page 28.")
