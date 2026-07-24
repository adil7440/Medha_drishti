import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.header("Data Augmentation Chain Viewer")

preprocessed_dir = Path(__file__).resolve().parent.parent.parent / "stage2" / "preprocessed"
npz_files = list(preprocessed_dir.glob("*.npz"))

if not npz_files:
    st.warning("No preprocessed data cache found. Please run stage2_pipeline.py first.")
    st.stop()

file_names = [f.name for f in npz_files]
selected_file = st.selectbox("Select Case for Data Augmentation Breakdown", options=file_names)

data = np.load(preprocessed_dir / selected_file)

aug_chain = {
    "Original": data["stage_final"],
    "Rotated (10°)": data["aug_rot"],
    "Flipped": data["aug_flip"],
    "Gamma Corrected": data["aug_gamma"],
    "Final Augmented": data["aug_final"]
}

st.markdown("Original $\\rightarrow$ Rotated $\\rightarrow$ Flipped $\\rightarrow$ Gamma Corrected $\\rightarrow$ Final Augmented")

cols = st.columns(5)
for idx, (title, img) in enumerate(aug_chain.items()):
    with cols[idx]:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
        ax.imshow(img, cmap='gray')
        ax.set_title(title, color='#f9fafb', fontsize=9, fontweight='600')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)
