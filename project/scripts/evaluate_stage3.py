import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.dncnn import DnCNN
from models.swinir import SwinIRSmall
from models.swinir_large import SwinIRLarge
from models.restormer import Restormer
from models.mirnet_v2 import MIRNetv2
from models.nafnet import NAFNet
from scripts.stage3_dataset import get_dataloaders
from scripts.quality_evaluator import QualityEvaluator
from scripts.train_stage3 import build_model, load_config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_usage() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    return 0.0


def evaluate_model_performance(model_name: str, model: torch.nn.Module,
                                val_loader, device, checkpoint_path: Path):
    model.eval()
    model = model.to(device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [Loaded] {model_name} checkpoint from: {checkpoint_path.name}")
    else:
        print(f"  [Warning] Checkpoint not found: {checkpoint_path}")

    metrics_accumulator = []
    inference_times = []

    with torch.no_grad():
        for inp, tgt, filenames in val_loader:
            inp, tgt = inp.to(device), tgt.to(device)

            start_t = time.time()
            pred = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_t_ms = (time.time() - start_t) * 1000.0 / inp.size(0)
            inference_times.append(infer_t_ms)

            pred_np = pred.cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            inp_np = inp.cpu().numpy()

            for b in range(inp.size(0)):
                orig_slice = inp_np[b, 0]
                proc_slice = pred_np[b, 0]

                eval_dict = QualityEvaluator.evaluate_pair(orig_slice, proc_slice)
                eval_dict["Model"] = model_name
                eval_dict["Filename"] = filenames[b]
                metrics_accumulator.append(eval_dict)

    df_eval = pd.DataFrame(metrics_accumulator)
    mean_metrics = df_eval.mean(numeric_only=True).to_dict()

    param_count = count_parameters(model)
    model_size_mb = round(param_count * 4 / (1024 ** 2), 2)
    mean_infer_ms = round(float(np.mean(inference_times)), 2)
    gpu_mem = get_gpu_memory_usage()

    summary = {
        "Model": model_name,
        "PSNR": round(mean_metrics.get("PSNR", 0.0), 4),
        "SSIM": round(mean_metrics.get("SSIM", 0.0), 4),
        "LPIPS": round(mean_metrics.get("LPIPS", 0.0), 4),
        "FSIM": round(mean_metrics.get("FSIM", 0.0), 4),
        "VIF": round(mean_metrics.get("VIF", 0.0), 4),
        "UQI": round(mean_metrics.get("UQI", 0.0), 4),
        "MSE": round(mean_metrics.get("MSE", 0.0), 6),
        "RMSE": round(mean_metrics.get("RMSE", 0.0), 4),
        "NIQE": round(mean_metrics.get("NIQE", 0.0), 4),
        "BRISQUE": round(mean_metrics.get("BRISQUE", 0.0), 4),
        "PIQE": round(mean_metrics.get("PIQE", 0.0), 4),
        "Entropy": round(mean_metrics.get("Entropy_After", 0.0), 4),
        "Contrast": round(mean_metrics.get("Contrast_After", 0.0), 4),
        "Sharpness": round(mean_metrics.get("Sharpness_After", 0.0), 4),
        "EdgeStrength": round(mean_metrics.get("EdgeStrength_After", 0.0), 4),
        "NoiseLevel": round(mean_metrics.get("NoiseLevel_After", 0.0), 4),
        "Inference_Time_ms": mean_infer_ms,
        "GPU_Memory_MB": gpu_mem,
        "Params": param_count,
        "Model_Size_MB": model_size_mb,
    }

    return summary, df_eval


def rank_models(leaderboard_df: pd.DataFrame, ranking_config: dict = None) -> pd.DataFrame:
    """Rank models using weighted composite score from config."""
    df = leaderboard_df.copy()

    if ranking_config is None:
        ranking_config = {
            "primary_metrics": [
                {"metric": "PSNR", "direction": "higher_better", "weight": 0.30},
                {"metric": "SSIM", "direction": "higher_better", "weight": 0.25},
                {"metric": "LPIPS", "direction": "lower_better", "weight": 0.15},
                {"metric": "NIQE", "direction": "lower_better", "weight": 0.15},
                {"metric": "BRISQUE", "direction": "lower_better", "weight": 0.15},
            ]
        }

    # Normalize metrics to [0, 1] for fair comparison
    for entry in ranking_config["primary_metrics"]:
        metric = entry["metric"]
        col = metric
        if col not in df.columns:
            continue

        vals = df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            df[f"_norm_{metric}"] = 0.5
        else:
            df[f"_norm_{metric}"] = (vals - vmin) / (vmax - vmin)

        if entry["direction"] == "lower_better":
            df[f"_norm_{metric}"] = 1.0 - df[f"_norm_{metric}"]

    # Compute composite score
    df["Composite_Score"] = 0.0
    for entry in ranking_config["primary_metrics"]:
        metric = entry["metric"]
        weight = entry["weight"]
        df["Composite_Score"] += weight * df[f"_norm_{metric}"]

    df = df.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    # Drop normalized helper columns
    norm_cols = [c for c in df.columns if c.startswith("_norm_")]
    df = df.drop(columns=norm_cols)

    return df


def run_stage3_evaluation(preprocessed_dir: str, config: dict = None):
    if config is None:
        config = load_config()

    print("=" * 80)
    print(" STAGE 3: AI-BASED MRI ENHANCEMENT MODEL BENCHMARKING & EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage3_dir = PROJECT_DIR / config.get("output", {}).get("base_dir", "stage3")
    metrics_dir = stage3_dir / "metrics"
    checkpoints_dir = stage3_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    bs = config.get("dataset", {}).get("batch_size", {})
    batch_size = bs.get("cuda", 4) if device.type == "cuda" else bs.get("cpu", 2)
    target_size = config.get("dataset", {}).get("target_size", 128)
    max_samples = config.get("dataset", {}).get("max_samples", None)

    _, val_loader = get_dataloaders(
        preprocessed_dir, batch_size=batch_size, target_size=target_size,
        max_samples=max_samples,
    )

    summaries = []
    all_dfs = {}

    models_cfg = config.get("models", {})
    for model_name, model_cfg in models_cfg.items():
        if not model_cfg.get("enabled", True):
            continue
        ckpt = checkpoints_dir / model_name.lower() / "best_checkpoint.pth"
        if not ckpt.exists():
            print(f"  [Skip] No checkpoint for {model_name}")
            continue

        print(f"\n>>> Evaluating {model_name}")
        try:
            model = build_model(model_name, model_cfg)
            summary, df_eval = evaluate_model_performance(
                model_name, model, val_loader, device, ckpt,
            )
            summaries.append(summary)
            all_dfs[model_name] = df_eval
        except Exception as e:
            print(f"  [Error] Evaluating {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not summaries:
        print("[Warning] No models evaluated.")
        return None

    leaderboard_df = pd.DataFrame(summaries)
    ranking_cfg = config.get("ranking", None)
    leaderboard_df = rank_models(leaderboard_df, ranking_cfg)

    csv_path = metrics_dir / "stage3_model_comparison.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n[Leaderboard Saved] {csv_path}")
    print(leaderboard_df[["Rank", "Model", "PSNR", "SSIM", "LPIPS", "NIQE", "BRISQUE", "Composite_Score"]].to_string(index=False))

    # Save per-model detailed metrics
    for model_name, df_eval in all_dfs.items():
        df_eval.to_csv(metrics_dir / f"{model_name.lower()}_detailed_metrics.csv", index=False)

    # Copy best model checkpoint
    best_model_name = leaderboard_df.iloc[0]["Model"]
    best_source = checkpoints_dir / best_model_name.lower() / "best_checkpoint.pth"
    best_dest = checkpoints_dir / "best_model.pth"
    if best_source.exists():
        shutil.copy(best_source, best_dest)
        print(f"\n[Best Model] {best_model_name} saved to: {best_dest}")

    return leaderboard_df


if __name__ == "__main__":
    prep_dir = PROJECT_DIR / "stage2" / "preprocessed"
    run_stage3_evaluation(str(prep_dir))
