import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cv2
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.dncnn import DnCNN
from models.swinir import SwinIRSmall
from models.swinir_large import SwinIRLarge
from models.restormer import Restormer
from models.mirnet_v2 import MIRNetv2
from models.nafnet import NAFNet
from scripts.train_stage3 import build_model, load_config
from scripts.quality_evaluator import QualityEvaluator

DARK_BG = '#0b0f19'
CARD_BG = '#111827'
GRID_COLOR = '#1f2937'
TEXT_COLOR = '#f9fafb'
ACCENT = '#0ea5e9'
GREEN = '#10b981'
RED = '#ef4444'
AMBER = '#f59e0b'
PURPLE = '#8b5cf6'


def _load_model(model_name: str, config: dict, checkpoints_dir: Path, device: torch.device):
    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        return None
    model = build_model(model_name, models_cfg[model_name])
    ckpt = checkpoints_dir / model_name.lower() / "best_checkpoint.pth"
    if ckpt.exists():
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def generate_comparison_visualizations(preprocessed_dir: str, config: dict = None):
    """Generate publication-quality comparison images for all models."""
    if config is None:
        config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage3_dir = PROJECT_DIR / config.get("output", {}).get("base_dir", "stage3")
    comp_dir = stage3_dir / "visualizations" / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = stage3_dir / "visualizations" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = stage3_dir / "checkpoints"

    prep_path = Path(preprocessed_dir)
    npz_files = sorted(prep_path.glob("*.npz"))
    if not npz_files:
        print("[Warning] No preprocessed files found.")
        return

    # Load enabled models
    models_cfg = config.get("models", {})
    enabled_models = [name for name, cfg in models_cfg.items() if cfg.get("enabled", True)]
    loaded_models = {}
    for name in enabled_models:
        if (checkpoints_dir / name.lower() / "best_checkpoint.pth").exists():
            model = _load_model(name, config, checkpoints_dir, device)
            if model is not None:
                loaded_models[name] = model

    if not loaded_models:
        print("[Warning] No trained models found.")
        return

    target_size = config.get("dataset", {}).get("target_size", 128)
    sample_files = npz_files[:min(10, len(npz_files))]

    for file_idx, npz_file in enumerate(sample_files):
        data = np.load(npz_file)
        orig = data["orig_slice"].astype(np.float32)
        target = data["stage_final"].astype(np.float32)

        # Normalize
        orig_min, orig_max = np.min(orig), np.max(orig)
        if orig_max > orig_min:
            orig = (orig - orig_min) / (orig_max - orig_min)
        tgt_min, tgt_max = np.min(target), np.max(target)
        if tgt_max > tgt_min:
            target = (target - tgt_min) / (tgt_max - tgt_min)

        orig_resized = cv2.resize(orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        target_resized = cv2.resize(target, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        n_models = len(loaded_models)
        fig = plt.figure(figsize=(4 * (n_models + 2), 10), facecolor=DARK_BG)
        gs = gridspec.GridSpec(3, n_models + 2, figure=fig, hspace=0.3, wspace=0.15)

        def _add_img(ax, img, title, cmap='gray'):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight='600', pad=8)
            ax.axis('off')

        # Row 1: Original, Preprocessed, Model outputs
        _add_img(fig.add_subplot(gs[0, 0]), orig_resized, "Original MRI")
        _add_img(fig.add_subplot(gs[0, 1]), target_resized, "Preprocessed MRI")

        for col_idx, (model_name, model) in enumerate(loaded_models.items()):
            inp_t = torch.from_numpy(target_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_t = model(inp_t)
            enhanced = pred_t.squeeze().cpu().numpy()
            enhanced = np.clip(enhanced, 0, 1)
            _add_img(fig.add_subplot(gs[0, col_idx + 2]), enhanced, f"{model_name}")

        # Row 2: Difference maps
        _add_img(fig.add_subplot(gs[1, 0]),
                 np.abs(orig_resized - target_resized),
                 "Preprocess Diff", cmap='inferno')
        _add_img(fig.add_subplot(gs[1, 1]),
                 np.abs(orig_resized - target_resized),
                 "Target Diff", cmap='inferno')

        for col_idx, (model_name, model) in enumerate(loaded_models.items()):
            inp_t = torch.from_numpy(target_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_t = model(inp_t)
            enhanced = np.clip(pred_t.squeeze().cpu().numpy(), 0, 1)
            diff_map = np.abs(target_resized - enhanced)
            _add_img(fig.add_subplot(gs[1, col_idx + 2]),
                     diff_map, f"{model_name} Diff", cmap='inferno')

        # Row 3: Residual heatmaps (enhanced - original)
        residual_orig = orig_resized - target_resized
        _add_img(fig.add_subplot(gs[2, 0]), residual_orig, "Residual (Orig-Proc)", cmap='RdBu_r')
        _add_img(fig.add_subplot(gs[2, 1]), residual_orig, "Residual", cmap='RdBu_r')

        for col_idx, (model_name, model) in enumerate(loaded_models.items()):
            inp_t = torch.from_numpy(target_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_t = model(inp_t)
            enhanced = np.clip(pred_t.squeeze().cpu().numpy(), 0, 1)
            residual = target_resized - enhanced
            _add_img(fig.add_subplot(gs[2, col_idx + 2]),
                     residual, f"{model_name} Residual", cmap='RdBu_r')

        fig.suptitle(
            f"Stage 3 MRI Enhancement Comparison — {npz_file.stem}",
            color=TEXT_COLOR, fontsize=14, fontweight='700', y=0.98
        )

        out_path = comp_dir / f"comparison_{file_idx:03d}_{npz_file.stem}.png"
        fig.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path.name}")

    print(f"[Done] {len(sample_files)} comparison images generated.")


def generate_edge_comparison(preprocessed_dir: str, config: dict = None):
    """Generate edge detection comparison images."""
    if config is None:
        config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage3_dir = PROJECT_DIR / config.get("output", {}).get("base_dir", "stage3")
    comp_dir = stage3_dir / "visualizations" / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = stage3_dir / "checkpoints"

    prep_path = Path(preprocessed_dir)
    npz_files = sorted(prep_path.glob("*.npz"))
    if not npz_files:
        return

    models_cfg = config.get("models", {})
    loaded_models = {}
    for name in [n for n, c in models_cfg.items() if c.get("enabled", True)]:
        if (checkpoints_dir / name.lower() / "best_checkpoint.pth").exists():
            model = _load_model(name, config, checkpoints_dir, device)
            if model:
                loaded_models[name] = model

    if not loaded_models:
        return

    target_size = config.get("dataset", {}).get("target_size", 128)

    for file_idx, npz_file in enumerate(npz_files[:5]):
        data = np.load(npz_file)
        orig = data["orig_slice"].astype(np.float32)
        target = data["stage_final"].astype(np.float32)

        o_min, o_max = np.min(orig), np.max(orig)
        if o_max > o_min:
            orig = (orig - o_min) / (o_max - o_min)
        t_min, t_max = np.min(target), np.max(target)
        if t_max > t_min:
            target = (target - t_min) / (t_max - t_min)

        orig_r = cv2.resize(orig, (target_size, target_size))
        target_r = cv2.resize(target, (target_size, target_size))

        n = len(loaded_models)
        fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8), facecolor=DARK_BG)

        # Row 1: Enhanced images
        axes[0, 0].imshow(target_r, cmap='gray')
        axes[0, 0].set_title("Preprocessed", color=TEXT_COLOR, fontsize=10, fontweight='600')
        axes[0, 0].axis('off')

        for col, (mname, model) in enumerate(loaded_models.items()):
            inp_t = torch.from_numpy(target_r).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = np.clip(model(inp_t).squeeze().cpu().numpy(), 0, 1)
            axes[0, col + 1].imshow(pred, cmap='gray')
            axes[0, col + 1].set_title(mname, color=TEXT_COLOR, fontsize=10, fontweight='600')
            axes[0, col + 1].axis('off')

        # Row 2: Edge detection
        edges_prep = cv2.Canny((target_r * 255).astype(np.uint8), 50, 150)
        axes[1, 0].imshow(edges_prep, cmap='gray')
        axes[1, 0].set_title("Edges (Preprocessed)", color=TEXT_COLOR, fontsize=10)
        axes[1, 0].axis('off')

        for col, (mname, model) in enumerate(loaded_models.items()):
            inp_t = torch.from_numpy(target_r).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = np.clip(model(inp_t).squeeze().cpu().numpy(), 0, 1)
            edges = cv2.Canny((pred * 255).astype(np.uint8), 50, 150)
            axes[1, col + 1].imshow(edges, cmap='gray')
            axes[1, col + 1].set_title(f"{mname} Edges", color=TEXT_COLOR, fontsize=10)
            axes[1, col + 1].axis('off')

        fig.suptitle(f"Edge Comparison — {npz_file.stem}", color=TEXT_COLOR, fontsize=13, fontweight='700')
        plt.tight_layout()
        out_path = comp_dir / f"edge_comparison_{file_idx:03d}.png"
        fig.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)

    print("[Done] Edge comparison images generated.")
