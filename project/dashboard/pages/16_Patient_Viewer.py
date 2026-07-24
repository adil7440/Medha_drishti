import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.quality_evaluator import QualityEvaluator

st.markdown("""
<div class="main-header">
    <h1 style="color:#06b6d4; margin:0 0 4px 0;">Patient Viewer</h1>
    <p>Browse preprocessed MRI patients and inspect per-pipeline-stage quality metrics.</p>
</div>
""", unsafe_allow_html=True)

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
npz_files = sorted(preprocessed_dir.glob("*.npz"))

if not npz_files:
    st.warning("No preprocessed data found.")
    st.stop()

# Group by dataset
brain_files = [f for f in npz_files if f.name.startswith("Brain")]
spine_files = [f for f in npz_files if f.name.startswith("Spine")]

dataset = st.radio("Dataset", ["Brain", "Spine", "All"], horizontal=True)
if dataset == "Brain":
    selected_files = brain_files
elif dataset == "Spine":
    selected_files = spine_files
else:
    selected_files = npz_files

file_names = [f.name for f in selected_files]
selected_file = st.selectbox("Select Patient Case", file_names)

data = np.load(preprocessed_dir / selected_file)
st.markdown(f"### Patient: `{selected_file}`")

# Available arrays in npz
available_keys = list(data.keys())
st.caption(f"Available arrays: {', '.join(available_keys)}")

# Display all stages
stage_keys = [k for k in available_keys if k not in ["volume_shape"]]
if not stage_keys:
    st.warning("No image arrays found in this file.")
    st.stop()

n_cols = min(4, len(stage_keys))
n_rows = (len(stage_keys) + n_cols - 1) // n_cols

for row_idx in range(n_rows):
    cols = st.columns(n_cols)
    for col_idx in range(n_cols):
        idx = row_idx * n_cols + col_idx
        if idx >= len(stage_keys):
            break
        key = stage_keys[idx]
        with cols[col_idx]:
            try:
                img = data[key].astype(np.float32)
                if img.ndim == 3:
                    img = img[:, :, img.shape[2] // 2]
                vmin, vmax = np.min(img), np.max(img)
                if vmax > vmin:
                    img = (img - vmin) / (vmax - vmin)
                fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
                ax.imshow(img, cmap='gray')
                ax.set_title(key, color='#f9fafb', fontsize=9, fontweight='600')
                ax.axis('off')
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.caption(f"Error loading {key}: {e}")

# Quality metrics for orig vs final
if "orig_slice" in data and "stage_final" in data:
    st.markdown("---")
    st.markdown("### Per-Case Quality Metrics")
    orig = data["orig_slice"].astype(np.float32)
    final = data["stage_final"].astype(np.float32)
    o_min, o_max = np.min(orig), np.max(orig)
    if o_max > o_min:
        orig = (orig - o_min) / (o_max - o_min)
    f_min, f_max = np.min(final), np.max(final)
    if f_max > f_min:
        final = (final - f_min) / (f_max - f_min)

    metrics = QualityEvaluator.evaluate_pair(orig, final)
    import pandas as pd
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ["Value"]
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
