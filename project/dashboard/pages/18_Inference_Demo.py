import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
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
from scripts.quality_evaluator import QualityEvaluator

st.set_page_config(page_title="Inference Demo", layout="wide")

st.markdown("""
<div class="main-header">
    <h1 style="color:#f97316; margin:0 0 4px 0;">Inference Demo</h1>
    <p>Run real-time AI inference on any MRI slice and see quality improvements live.</p>
</div>
""", unsafe_allow_html=True)

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
checkpoints_dir = PROJECT_DIR / "stage3" / "checkpoints"

npz_files = sorted(preprocessed_dir.glob("*.npz"))
if not npz_files:
    st.warning("No preprocessed data found.")
    st.stop()

MODEL_REGISTRY = {
    "DnCNN": (lambda: DnCNN(in_channels=1, out_channels=1, num_features=96, num_layers=17), "Baseline CNN"),
    "SwinIR_Small": (lambda: SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8), "Swin Transformer"),
    "SwinIR_Large": (lambda: SwinIRLarge(in_channels=1, out_channels=1, embed_dim=180, num_heads=6, window_size=8), "Swin Transformer"),
    "Restormer": (lambda: Restormer(in_channels=1, out_channels=1, dim=48), "Linear Attention"),
    "MIRNet_v2": (lambda: MIRNetv2(in_channels=1, out_channels=1), "Multi-Scale ResNet"),
    "NAFNet": (lambda: NAFNet(in_channels=1, out_channels=1, width=32), "Activation-Free"),
}

available = []
for name, (factory, desc) in MODEL_REGISTRY.items():
    if (checkpoints_dir / name.lower() / "best_checkpoint.pth").exists():
        available.append((name, desc))

if not available:
    st.info("No trained models found. Run `run_stage3.py` first.")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c1, c2 = st.columns(2)
with c1:
    selected_file = st.selectbox("Select MRI Case", [f.name for f in npz_files])
with c2:
    model_option = st.selectbox("Select Model", [f"{n} ({d})" for n, d in available])
    model_name = model_option.split(" (")[0]

# Load model
factory = MODEL_REGISTRY[model_name][0]
model = factory()
ckpt = checkpoints_dir / model_name.lower() / "best_checkpoint.pth"
if ckpt.exists():
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
model = model.to(device).eval()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = round(param_count * 4 / (1024 ** 2), 2)

# Load data
data = np.load(preprocessed_dir / selected_file)
orig = data["orig_slice"].astype(np.float32)
target = data["stage_final"].astype(np.float32)

for arr_name in ["orig", "target"]:
    arr = locals()[arr_name]
    vmin, vmax = np.min(arr), np.max(arr)
    if vmax > vmin:
        locals()[arr_name] = (arr - vmin) / (vmax - vmin)

orig_r = cv2.resize(orig, (128, 128))
target_r = cv2.resize(target, (128, 128))

# Run inference
st.markdown("### Running Inference...")
inp_t = torch.from_numpy(target_r).unsqueeze(0).unsqueeze(0).float().to(device)

start_time = time.time()
with torch.no_grad():
    enhanced = model(inp_t)
if device.type == "cuda":
    torch.cuda.synchronize()
infer_time = (time.time() - start_time) * 1000

enhanced_np = np.clip(enhanced.squeeze().cpu().numpy(), 0, 1)

# Metrics
metrics = QualityEvaluator.evaluate_pair(target_r, enhanced_np)

# Display results
st.markdown("### Inference Results")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Inference Time", f"{infer_time:.1f} ms")
with c2:
    st.metric("PSNR", f"{metrics['PSNR']:.4f} dB")
with c3:
    st.metric("SSIM", f"{metrics['SSIM']:.4f}")
with c4:
    st.metric("Model Size", f"{model_size_mb:.2f} MB")
with c5:
    st.metric("Parameters", f"{param_count:,}")

# Side-by-side comparison
st.markdown("### Visual Comparison")
c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    ax.imshow(orig_r, cmap='gray')
    ax.set_title("Original MRI", color='#f9fafb', fontsize=11, fontweight='600')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

with c2:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    ax.imshow(target_r, cmap='gray')
    ax.set_title("Preprocessed (Input)", color='#f9fafb', fontsize=11, fontweight='600')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

with c3:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    ax.imshow(enhanced_np, cmap='gray')
    ax.set_title(f"{model_name} Enhanced", color='#10b981', fontsize=11, fontweight='600')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

# Difference maps
st.markdown("### Difference Analysis")
c1, c2, c3 = st.columns(3)

diff = np.abs(target_r - enhanced_np)
residual = target_r - enhanced_np

with c1:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    im = ax.imshow(diff, cmap='inferno')
    ax.set_title("|Preprocessed - Enhanced|", color='#f9fafb', fontsize=10, fontweight='600')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)

with c2:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    im = ax.imshow(residual, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title("Signed Residual", color='#f9fafb', fontsize=10, fontweight='600')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)

with c3:
    orig_edges = cv2.Canny((orig_r * 255).astype(np.uint8), 50, 150)
    enh_edges = cv2.Canny((enhanced_np * 255).astype(np.uint8), 50, 150)
    edge_diff = np.abs(orig_edges.astype(float) - enh_edges.astype(float))
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0b0f19')
    im = ax.imshow(edge_diff, cmap='hot')
    ax.set_title("Edge Difference", color='#f9fafb', fontsize=10, fontweight='600')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)

# Full metrics
st.markdown("---")
st.markdown("### Full Quality Metrics")
import pandas as pd
metrics_df = pd.DataFrame([metrics]).T
metrics_df.columns = ["Value"]
st.dataframe(metrics_df.style.format("{:.6f}"), use_container_width=True)
