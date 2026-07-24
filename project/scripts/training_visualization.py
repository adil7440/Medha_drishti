import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DARK_BG = '#0b0f19'
CARD_BG = '#111827'
TEXT_COLOR = '#f9fafb'
ACCENT = '#0ea5e9'
GREEN = '#10b981'
RED = '#ef4444'
AMBER = '#f59e0b'
PURPLE = '#8b5cf6'
GRID_COLOR = '#374151'

def generate_all_training_charts(stage3_dir: Path):
    """Generate all training visualization charts for every trained model."""
    logs_dir = stage3_dir / "logs"
    charts_dir = stage3_dir / "visualizations" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    history_files = list(logs_dir.glob("*_history.json"))
    if not history_files:
        print("[Warning] No training history files found.")
        return

    for hist_file in history_files:
        with open(hist_file, "r") as f:
            data = json.load(f)

        model_name = data.get("model_name", hist_file.stem.replace("_history", ""))
        history = data.get("history", [])
        total_time = data.get("total_train_time_sec", 0)
        if not history:
            continue

        df = pd.DataFrame(history)
        epochs = df["epoch"]

        # 1. Training & Validation Loss Curves
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.plot(epochs, df["train_loss"], color=ACCENT, linewidth=2, label="Train Loss", marker='o', markersize=3)
        ax.plot(epochs, df["val_loss"], color=RED, linewidth=2, label="Val Loss", marker='s', markersize=3)
        ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel("Loss", color=TEXT_COLOR, fontsize=12)
        ax.set_title(f"{model_name} — Training & Validation Loss", color=TEXT_COLOR, fontsize=14, fontweight='700')
        ax.legend(facecolor=CARD_BG, edgecolor='#374151', labelcolor=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        fig.savefig(charts_dir / f"{model_name.lower()}_loss_curve.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # 2. Validation PSNR Curve
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.plot(epochs, df["val_psnr"], color=GREEN, linewidth=2.5, marker='D', markersize=4)
        ax.fill_between(epochs, df["val_psnr"], alpha=0.15, color=GREEN)
        ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel("PSNR (dB)", color=TEXT_COLOR, fontsize=12)
        ax.set_title(f"{model_name} — Validation PSNR", color=TEXT_COLOR, fontsize=14, fontweight='700')
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        fig.savefig(charts_dir / f"{model_name.lower()}_psnr_curve.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # 3. Learning Rate Schedule
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.plot(epochs, df["learning_rate"], color=AMBER, linewidth=2)
        ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel("Learning Rate", color=TEXT_COLOR, fontsize=12)
        ax.set_title(f"{model_name} — Learning Rate Schedule", color=TEXT_COLOR, fontsize=14, fontweight='700')
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_yscale('log')
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        fig.savefig(charts_dir / f"{model_name.lower()}_lr_schedule.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)

        # 4. GPU Memory & Epoch Time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
        for ax in [ax1, ax2]:
            ax.set_facecolor(DARK_BG)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            ax.tick_params(colors=TEXT_COLOR)

        ax1.plot(epochs, df["gpu_mem_mb"], color=PURPLE, linewidth=2, marker='^', markersize=4)
        ax1.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=11)
        ax1.set_ylabel("GPU Memory (MB)", color=TEXT_COLOR, fontsize=11)
        ax1.set_title("GPU Memory Usage", color=TEXT_COLOR, fontsize=13, fontweight='600')

        ax2.plot(epochs, df["epoch_time_sec"], color=RED, linewidth=2, marker='v', markersize=4)
        ax2.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=11)
        ax2.set_ylabel("Time (s)", color=TEXT_COLOR, fontsize=11)
        ax2.set_title("Epoch Training Time", color=TEXT_COLOR, fontsize=13, fontweight='600')

        fig.suptitle(f"{model_name} — Infrastructure Metrics", color=TEXT_COLOR, fontsize=14, fontweight='700', y=1.02)
        plt.tight_layout()
        fig.savefig(charts_dir / f"{model_name.lower()}_infra_metrics.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
        plt.close(fig)

    print(f"[Done] Training charts generated for {len(history_files)} models.")


def generate_model_comparison_charts(metrics_csv_path: str, output_dir: Path):
    """Generate leaderboard comparison charts."""
    if not Path(metrics_csv_path).exists():
        return

    df = pd.read_csv(metrics_csv_path)
    charts_dir = output_dir / "visualizations" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    models = df["Model"].tolist()
    n = len(models)
    colors = [ACCENT, GREEN, RED, AMBER, PURPLE, '#06b6d4'][:n]

    # 1. PSNR & SSIM Bar Charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    bars1 = ax1.bar(models, df["PSNR"], color=colors, edgecolor='#374151', linewidth=1)
    ax1.set_ylabel("PSNR (dB)", color=TEXT_COLOR, fontsize=12)
    ax1.set_title("PSNR Comparison", color=TEXT_COLOR, fontsize=14, fontweight='700')
    ax1.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars1, df["PSNR"]):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10, fontweight='600')

    bars2 = ax2.bar(models, df["SSIM"], color=colors, edgecolor='#374151', linewidth=1)
    ax2.set_ylabel("SSIM", color=TEXT_COLOR, fontsize=12)
    ax2.set_title("SSIM Comparison", color=TEXT_COLOR, fontsize=14, fontweight='700')
    ax2.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars2, df["SSIM"]):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=10, fontweight='600')

    plt.tight_layout()
    fig.savefig(charts_dir / "leaderboard_psnr_ssim.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)

    # 2. Radar Chart (Top 3 metrics)
    radar_metrics = ["PSNR", "SSIM", "FSIM"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, row in df.iterrows():
        values = [row[m] for m in radar_metrics]
        # Normalize to [0, 1] relative to column max
        max_vals = [df[m].max() for m in radar_metrics]
        norm_vals = [v / (mx + 1e-8) for v, mx in zip(values, max_vals)]
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, 'o-', linewidth=2, label=row["Model"], color=colors[i % len(colors)])
        ax.fill(angles, norm_vals, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, color=TEXT_COLOR, fontsize=11)
    ax.set_title("Model Performance Radar", color=TEXT_COLOR, fontsize=14, fontweight='700', pad=20)
    ax.legend(facecolor=CARD_BG, edgecolor='#374151', labelcolor=TEXT_COLOR, fontsize=10, loc='upper right')
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(color=GRID_COLOR)

    fig.savefig(charts_dir / "leaderboard_radar.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)

    # 3. Inference Speed & Model Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    bars1 = ax1.barh(models, df["Inference_Time_ms"], color=colors, edgecolor='#374151')
    ax1.set_xlabel("Inference Time (ms)", color=TEXT_COLOR, fontsize=12)
    ax1.set_title("Inference Speed", color=TEXT_COLOR, fontsize=14, fontweight='700')
    for bar, val in zip(bars1, df["Inference_Time_ms"]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.1f}ms', va='center', color=TEXT_COLOR, fontsize=10)

    bars2 = ax2.barh(models, df["Model_Size_MB"], color=colors, edgecolor='#374151')
    ax2.set_xlabel("Model Size (MB)", color=TEXT_COLOR, fontsize=12)
    ax2.set_title("Model Size", color=TEXT_COLOR, fontsize=14, fontweight='700')
    for bar, val in zip(bars2, df["Model_Size_MB"]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{val:.2f}MB', va='center', color=TEXT_COLOR, fontsize=10)

    plt.tight_layout()
    fig.savefig(charts_dir / "leaderboard_speed_size.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)

    # 4. Composite Score
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    bars = ax.bar(models, df["Composite_Score"], color=colors, edgecolor='#374151', linewidth=1)
    ax.set_ylabel("Composite Score", color=TEXT_COLOR, fontsize=12)
    ax.set_title("Overall Model Ranking (Composite Score)", color=TEXT_COLOR, fontsize=14, fontweight='700')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, df["Composite_Score"]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=11, fontweight='700')
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    fig.savefig(charts_dir / "leaderboard_composite.png", dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)

    print(f"[Done] Model comparison charts generated.")
