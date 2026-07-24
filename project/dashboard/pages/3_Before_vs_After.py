import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dashboard"))

from components.image_comparison import display_side_by_side

st.header("Before vs After Preprocessing Viewer")

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
npz_files = list(preprocessed_dir.glob("*.npz"))

if not npz_files:
    st.warning("No preprocessed data cache found. Please run stage2_pipeline.py first.")
    st.stop()

file_names = [f.name for f in npz_files]
selected_file = st.selectbox("Select Processed MRI Case", options=file_names)

data = np.load(preprocessed_dir / selected_file)
orig_slice = data["orig_slice"]
final_slice = data["stage_final"]

st.markdown("### Side-by-Side Slice Comparison")
display_side_by_side(orig_slice, final_slice, label1="Original Raw MRI", label2="Preprocessed & Enhanced MRI")
