import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.header("Interactive Slice Viewer")

preprocessed_dir = Path(__file__).resolve().parent.parent.parent / "stage2" / "preprocessed"
npz_files = list(preprocessed_dir.glob("*.npz"))

if not npz_files:
    st.warning("No preprocessed data cache found. Please run stage2_pipeline.py first.")
    st.stop()

file_names = [f.name for f in npz_files]
selected_file = st.selectbox("Select Volume Case", options=file_names)

data = np.load(preprocessed_dir / selected_file)
final_slice = data["stage_final"]

# Simulate multi-slice view around central slice
slices = []
for factor in [0.7, 0.85, 1.0, 1.15, 1.3]:
    slices.append(np.clip(final_slice * factor, 0, 1))

slice_idx = st.slider("Move through Slices (Z-Axis)", min_value=1, max_value=len(slices), value=3)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0b0f19')
ax.imshow(slices[slice_idx - 1], cmap='gray')
ax.set_title(f"Slice {slice_idx} of {len(slices)}", color='#f9fafb', fontsize=12)
ax.axis('off')
st.pyplot(fig, clear_figure=True)
