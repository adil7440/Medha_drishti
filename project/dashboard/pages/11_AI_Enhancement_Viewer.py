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

st.set_page_config(page_title="AI Enhancement Viewer", layout="wide")

st.markdown("""
<div class="main-header">
    <h1 style="color:#10b981; margin:0 0 4px 0;">AI MRI Enhancement Viewer</h1>
    <p>Compare AI-enhanced outputs across all trained models side-by-side.</p>
</div>
""", unsafe_allow_html=True)

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
checkpoints_dir = PROJECT_DIR / "stage3" / "checkpoints"

npz_files = sorted(preprocessed_dir.glob("*.npz"))
if not npz_files:
    st.warning("No preprocessed MRI cases found. Run Stage 2 first.")
    st.stop()

MODEL_REGISTRY = {
    "DnCNN": lambda: DnCNN(in_channels=1, out_channels=1, num_features=96, num_layers=17),
    "SwinIR_Small": lambda: SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8),
    "SwinIR_Large": lambda: SwinIRLarge(in_channels=1, out_channels=1, embed_dim=180, num_heads=6, window_size=8),
    "Restormer": lambda: Restormer(in_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8]),
    "MIRNet_v2": lambda: MIRNetv2(in_channels=1, out_channels=1, features=[32, 64, 128]),
    "NAFNet": lambda: NAFNet(in_channels=1, out_channels=1, width=32, middle_blk_num=1),
}

available_models = []
for name, factory in MODEL_REGISTRY.items():
    ckpt = checkpoints_dir / name.lower() / "best_checkpoint.pth"
    if ckpt.exists():
        available_models.append(name)

if not available_models:
    st.warning("No trained models found. Run `run_stage3.py` first.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    selected_file = st.selectbox("Select Patient / MRI Case", [f.name for f in npz_files])
with c2:
    selected_models = st.multiselect("Select Models", available_models, default=available_models[:3])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(preprocessed_dir / selected_file)
orig = data["orig_slice"].astype(np.float32)
target = data["stage_final"].astype(np.float32)

o_min, o_max = np.min(orig), np.max(orig)
if o_max > o_min:
    orig = (orig - o_min) / (o_max - o_min)
t_min, t_max = np.min(target), np.max(target)
if t_max > t_min:
    target = (target - t_min) / (t_max - t_min)

orig_r = cv2.resize(orig, (128, 128))
target_r = cv2.resize(target, (128, 128))

enhanced_dict = {}
for model_name in selected_models:
    model = MODEL_REGISTRY[model_name]()
    ckpt = checkpoints_dir / model_name.lower() / "best_checkpoint.pth"
    if ckpt.exists():
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
    model = model.to(device).eval()
    inp_t = torch.from_numpy(target_r).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(inp_t)
    enhanced_dict[model_name] = np.clip(pred.squeeze().cpu().numpy(), 0, 1)

# Row 1: Original + Preprocessed
st.markdown("### Visual Progression")
col_count = 2 + len(selected_models)
cols = st.columns(col_count)

with cols[0]:
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
    ax.imshow(orig_r, cmap='gray')
    ax.set_title("Original", color='#f9fafb', fontsize=10, fontweight='600')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

with cols[1]:
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
    ax.imshow(target_r, cmap='gray')
    ax.set_title("Preprocessed", color='#f9fafb', fontsize=10, fontweight='600')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

for i, (mname, enhanced) in enumerate(enhanced_dict.items()):
    with cols[i + 2]:
        fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
        ax.imshow(enhanced, cmap='gray')
        ax.set_title(mname, color='#f9fafb', fontsize=10, fontweight='600')
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)

# Row 2: Difference Maps
st.markdown("### Difference Maps (|Preprocessed - Enhanced|)")
cols2 = st.columns(len(selected_models) if selected_models else 1)
for i, (mname, enhanced) in enumerate(enhanced_dict.items()):
    with cols2[i]:
        diff = np.abs(target_r - enhanced)
        fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
        im = ax.imshow(diff, cmap='inferno')
        ax.set_title(f"{mname} Diff", color='#f9fafb', fontsize=10, fontweight='600')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)

# Row 3: Residual Heatmaps
st.markdown("### Residual Heatmaps (Original - Enhanced)")
cols3 = st.columns(len(selected_models) if selected_models else 1)
for i, (mname, enhanced) in enumerate(enhanced_dict.items()):
    with cols3[i]:
        residual = orig_r - enhanced
        fig, ax = plt.subplots(figsize=(3, 3), facecolor='#0b0f19')
        im = ax.imshow(residual, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{mname} Residual", color='#f9fafb', fontsize=10, fontweight='600')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)

# Model Metrics Summary
if selected_models:
    st.markdown("### Per-Model Metrics")
    from scripts.quality_evaluator import QualityEvaluator
    metrics_data = []
    for mname, enhanced in enhanced_dict.items():
        m = QualityEvaluator.evaluate_pair(target_r, enhanced)
        m["Model"] = mname
        metrics_data.append(m)
    import pandas as pd
    st.dataframe(pd.DataFrame(metrics_data)[["Model", "PSNR", "SSIM", "LPIPS", "FSIM", "RMSE", "NIQE", "BRISQUE"]].round(4), use_container_width=True)
