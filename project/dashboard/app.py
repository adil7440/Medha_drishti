import os
import sys
from pathlib import Path
import streamlit as st

# Setup paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

st.set_page_config(
    page_title="MRI Preprocessing Workstation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = PROJECT_DIR / "dashboard" / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Header
st.sidebar.markdown(
    """
    <div style="padding: 10px 0;">
        <div class="sidebar-title">MedhaDrishti AI</div>
        <div class="sidebar-sub">Stage 2: MRI Preprocessing Pipeline</div>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
        <h1>MRI Preprocessing Workstation</h1>
        <p>
            Standardized medical image preprocessing, bias field correction, noise reduction, 
            contrast optimization, and quantitative metric analysis for Brain and Spine MRI datasets.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick Overview Columns
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Brain Dataset</div>
        <div class="metric-value">369 Cases</div>
        <div class="metric-sub">BraTS 2020 (T1, T1CE, T2, FLAIR)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Spine Dataset</div>
        <div class="metric-value">10 Cases</div>
        <div class="metric-sub">Normal & Pathological Spine</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Quality Metrics</div>
        <div class="metric-value">17 Metrics</div>
        <div class="metric-sub">PSNR, SSIM, Entropy, Noise</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Pipeline Architecture</div>
        <div class="metric-value">Classical</div>
        <div class="metric-sub">Deterministic Signal Processing</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("Use the sidebar navigation to view patient data, pipeline stages, metrics, and technical reports.")
