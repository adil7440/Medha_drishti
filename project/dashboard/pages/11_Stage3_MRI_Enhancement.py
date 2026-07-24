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

st.header("Stage 3 - AI MRI Image Enhancement")

preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
checkpoints_dir = PROJECT_DIR / "stage3" / "checkpoints"

npz_files = list(preprocessed_dir.glob("*.npz"))
if not npz_files:
    st.warning("No preprocessed MRI cases found. Please run Stage 2 pipeline first.")
    st.stop()

file_names = [f.name for f in npz_files]

c1, c2, c3 = st.columns(3)
with c1:
    selected_file = st.selectbox("Select Patient / MRI Case", options=file_names)
with c2:
    model_choice = st.selectbox("Select AI Model", options=["SwinIR Small (Primary)", "DnCNN (Baseline)"])

data = np.load(preprocessed_dir / selected_file)
orig_slice = data["orig_slice"]
prep_slice = data["stage_final"]

# Scale normalize
orig_norm = (orig_slice - np.min(orig_slice)) / (np.max(orig_slice) - np.min(orig_slice) + 1e-8)
prep_norm = (prep_slice - np.min(prep_slice)) / (np.max(prep_slice) - np.min(prep_slice) + 1e-8)

# Resize to standard model resolution (128, 128)
prep_resized = cv2.resize(prep_norm, (128, 128), interpolation=cv2.INTER_LINEAR)
orig_resized = cv2.resize(orig_norm, (128, 128), interpolation=cv2.INTER_LINEAR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "SwinIR" in model_choice:
    model = SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8)
    ckpt_p = checkpoints_dir / "swinir" / "best_checkpoint.pth"
else:
    model = DnCNN(in_channels=1, out_channels=1, num_features=64, num_layers=17)
    ckpt_p = checkpoints_dir / "dncnn" / "best_checkpoint.pth"

model = model.to(device)
if ckpt_p.exists():
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    st.caption(f"Loaded {model_choice} checkpoint from: {ckpt_p.name}")
else:
    st.info("Using initialized model weights (run `run_stage3.py` to train models on GPU).")

model.eval()

# Run AI Inference
inp_t = torch.from_numpy(prep_resized).unsqueeze(0).unsqueeze(0).float().to(device)
with torch.no_grad():
    enhanced_t = model(inp_t)
enhanced_slice = enhanced_t.squeeze().cpu().numpy()
enhanced_slice = np.clip(enhanced_slice, 0, 1)

diff_map = np.abs(prep_resized - enhanced_slice)

st.markdown("### AI Enhancement Visual Progression")
st.markdown("Original MRI $\\rightarrow$ Preprocessed MRI $\\rightarrow$ AI Enhanced Output $\\rightarrow$ Difference Map")

fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0b0f19')

axes[0].imshow(orig_resized, cmap='gray')
axes[0].set_title("1. Original Raw MRI", color='#f9fafb', fontsize=11, fontweight='600')
axes[0].axis('off')

axes[1].imshow(prep_resized, cmap='gray')
axes[1].set_title("2. Preprocessed MRI", color='#f9fafb', fontsize=11, fontweight='600')
axes[1].axis('off')

axes[2].imshow(enhanced_slice, cmap='gray')
axes[2].set_title(f"3. {model_choice} Output", color='#f9fafb', fontsize=11, fontweight='600')
axes[2].axis('off')

im = axes[3].imshow(diff_map, cmap='inferno')
axes[3].set_title("4. AI Difference Map", color='#f9fafb', fontsize=11, fontweight='600')
axes[3].axis('off')

plt.tight_layout()
st.pyplot(fig, clear_figure=True)
