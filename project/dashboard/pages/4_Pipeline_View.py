import streamlit as st
import numpy as np
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.image_comparison import display_stage_grid

st.header("Sequential Preprocessing Pipeline Inspector")

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
npz_files = list(preprocessed_dir.glob("*.npz"))

if not npz_files:
    st.warning("No preprocessed data cache found. Please run stage2_pipeline.py first.")
    st.stop()

file_names = [f.name for f in npz_files]
selected_file = st.selectbox("Select MRI Case to Inspect Pipeline Stages", options=file_names)

data = np.load(preprocessed_dir / selected_file)

stages = {
    "1. Original": data["orig_slice"],
    "2. Normalization": data["stage_norm"],
    "3. Denoising": data["stage_denoise_bilat"],
    "4. N4 Bias Correction": data["stage_n4"],
    "5. CLAHE Contrast": data["stage_clahe"],
    "6. Final Output": data["stage_final"]
}

st.markdown("### Pipeline Execution Steps")
st.markdown("Original $\\rightarrow$ Normalization $\\rightarrow$ Denoising $\\rightarrow$ N4 Bias Correction $\\rightarrow$ CLAHE $\\rightarrow$ Final Output")

display_stage_grid(stages)

st.markdown("---")
stage_key = st.radio("Select Stage for Full View", list(stages.keys()))
st.image(stages[stage_key], caption=stage_key, use_container_width=True, clamp=True)
