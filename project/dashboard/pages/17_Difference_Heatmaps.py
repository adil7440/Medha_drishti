import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.dncnn import DnCNN
from models.swinir import SwinIRSmall
from models.swinir_large import SwinIRLarge
from models.restormer import Restormer
from models.mirnet_v2 import MIRNetv2
from models.nafnet import NAFNet

st.set_page_config(page_title="Difference Heatmaps", layout="wide")

st.markdown("""
<div class="main-header">
    <h1 style="color:#ef4444; margin:0 0 4px 0;">Difference Heatmaps</h1>
    <p>Visualize pixel-level differences between original, preprocessed, and AI-enhanced MRI outputs.</p>
</div>
""", unsafe_allow_html=True)

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
checkpoints_dir = PROJECT_DIR / "stage3" / "checkpoints"

npz_files = sorted(preprocessed_dir.glob("*.npz"))
if not npz_files:
    st.warning("No preprocessed data found.")
    st.stop()

MODEL_REGISTRY = {
    "DnCNN": lambda: DnCNN(in_channels=1, out_channels=1, num_features=96, num_layers=17),
    "SwinIR_Small": lambda: SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8),
    "SwinIR_Large": lambda: SwinIRLarge(in_channels=1, out_channels=1, embed_dim=180, num_heads=6, window_size=8),
    "Restormer": lambda: Restormer(in_channels=1, out_channels=1, dim=48),
    "MIRNet_v2": lambda: MIRNetv2(in_channels=1, out_channels=1),
    "NAFNet": lambda: NAFNet(in_channels=1, out_channels=1, width=32),
}

available = [n for n in MODEL_REGISTRY if (checkpoints_dir / n.lower() / "best_checkpoint.pth").exists()]

c1, c2 = st.columns([2, 1])
with c1:
    selected_file = st.selectbox("Select Patient", [f.name for f in npz_files])
with c2:
    model_name = st.selectbox("Select Model", available if available else ["None"])

if not available:
    st.info("No trained models available.")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(preprocessed_dir / selected_file)
orig = data["orig_slice"].astype(np.float32)
target = data["stage_final"].astype(np.float32)

for arr, name in [(orig, "orig"), (target, "target")]:
    vmin, vmax = np.min(arr), np.max(arr)
    if vmax > vmin:
        exec(f"{name} = ({name} - vmin) / (vmax - vmin)")

orig_r = cv2.resize(orig, (128, 128))
target_r = cv2.resize(target, (128, 128))

# Run inference
model = MODEL_REGISTRY[model_name]()
ckpt = checkpoints_dir / model_name.lower() / "best_checkpoint.pth"
if ckpt.exists():
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
model = model.to(device).eval()

inp_t = torch.from_numpy(target_r).unsqueeze(0).unsqueeze(0).float().to(device)
with torch.no_grad():
    enhanced = np.clip(model(inp_t).squeeze().cpu().numpy(), 0, 1)

# Generate difference maps
diff_prep = np.abs(orig_r - target_r)
diff_enhanced = np.abs(orig_r - enhanced)
diff_residual = target_r - enhanced
abs_diff = np.abs(target_r - enhanced)

# Visualization
st.markdown("### Heatmap Comparison")

# Row 1: Source images
st.markdown("**Source Images**")
c1, c2, c3 = st.columns(3)
for col, (img, title) in zip([c1, c2, c3], [(orig_r, "Original"), (target_r, "Preprocessed"), (enhanced, f"Enhanced ({model_name})")]):
    with col:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
        ax.imshow(img, cmap='gray')
        ax.set_title(title, color='#f9fafb', fontsize=11, fontweight='600')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)

# Row 2: Absolute difference heatmaps
st.markdown("**Absolute Difference Maps**")
c1, c2, c3 = st.columns(3)
for col, (img, title) in zip([c1, c2, c3],
    [(diff_prep, "|Original - Preprocessed|"),
     (diff_enhanced, "|Original - Enhanced|"),
     (abs_diff, "|Preprocessed - Enhanced|")]):
    with col:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
        im = ax.imshow(img, cmap='inferno', vmin=0, vmax=max(img.max(), 0.01))
        ax.set_title(title, color='#f9fafb', fontsize=11, fontweight='600')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        st.pyplot(fig, clear_figure=True)

# Row 3: Signed residual heatmaps
st.markdown("**Signed Residual Heatmaps (Red = brighter, Blue = darker)**")
c1, c2 = st.columns(2)
for col, (img, title) in zip([c1, c2],
    [(diff_residual, "Preprocessed - Enhanced"),
     (orig_r - enhanced, "Original - Enhanced")]):
    with col:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0b0f19')
        im = ax.imshow(img, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(title, color='#f9fafb', fontsize=11, fontweight='600')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Signed Diff')
        st.pyplot(fig, clear_figure=True)

# Row 4: Edge comparison
st.markdown("**Edge Detection Comparison**")
c1, c2, c3 = st.columns(3)
for col, (img, title) in zip([c1, c2, c3],
    [(orig_r, "Original Edges"), (target_r, "Preprocessed Edges"), (enhanced, f"{model_name} Edges")]):
    with col:
        edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
        ax.imshow(edges, cmap='gray')
        ax.set_title(title, color='#f9fafb', fontsize=11, fontweight='600')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)

# Statistics
st.markdown("---")
st.markdown("### Difference Statistics")
import pandas as pd
stats = {
    "Metric": ["Mean Abs Diff", "Max Abs Diff", "Std Diff", "PSNR", "SSIM"],
    "Original vs Preprocessed": [
        f"{diff_prep.mean():.6f}", f"{diff_prep.max():.6f}", f"{diff_prep.std():.6f}",
        f"{20 * np.log10(1.0 / np.sqrt(np.mean((orig_r - target_r)**2) + 1e-10)):.2f} dB",
        "—"
    ],
    f"Original vs {model_name}": [
        f"{diff_enhanced.mean():.6f}", f"{diff_enhanced.max():.6f}", f"{diff_enhanced.std():.6f}",
        f"{20 * np.log10(1.0 / np.sqrt(np.mean((orig_r - enhanced)**2) + 1e-10)):.2f} dB",
        "—"
    ],
}
st.dataframe(pd.DataFrame(stats), use_container_width=True)
