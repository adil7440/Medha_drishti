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
from scripts.stage3_dataset import get_dataloaders
from scripts.quality_evaluator import QualityEvaluator


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_performance(model_name: str, model: torch.nn.Module, val_loader, device, checkpoint_path: Path):
    model.eval()
    model = model.to(device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Loaded] Checkpoint loaded for {model_name} from: {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint not found at: {checkpoint_path}")

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

    summary = {
        "Model": model_name,
        "PSNR": round(mean_metrics.get("PSNR", 0.0), 2),
        "SSIM": round(mean_metrics.get("SSIM", 0.0), 4),
        "LPIPS": round(mean_metrics.get("LPIPS", 0.0), 4),
        "MSE": round(mean_metrics.get("MSE", 0.0), 6),
        "RMSE": round(mean_metrics.get("RMSE", 0.0), 4),
        "UQI": round(mean_metrics.get("UQI", 0.0), 4),
        "FSIM": round(mean_metrics.get("FSIM", 0.0), 4),
        "BRISQUE": round(mean_metrics.get("BRISQUE", 0.0), 2),
        "Contrast_After": round(mean_metrics.get("Contrast_After", 0.0), 4),
        "NoiseLevel_After": round(mean_metrics.get("NoiseLevel_After", 0.0), 4),
        "Inference_Time_ms": mean_infer_ms,
        "Params": param_count,
        "Model_Size_MB": model_size_mb
    }

    return summary, df_eval


def run_stage3_evaluation(preprocessed_dir: str):
    print("=" * 80)
    print(" STAGE 3: AI-BASED MRI ENHANCEMENT MODEL BENCHMARKING & EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage3_dir = PROJECT_DIR / "stage3"
    metrics_dir = stage3_dir / "metrics"
    checkpoints_dir = stage3_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = get_dataloaders(preprocessed_dir, batch_size=4 if device.type == "cuda" else 2, target_size=128, max_samples=40)

    # 1. Evaluate DnCNN
    dncnn = DnCNN(in_channels=1, out_channels=1, num_features=64, num_layers=17)
    dncnn_ckpt = checkpoints_dir / "dncnn" / "best_checkpoint.pth"
    dncnn_summary, dncnn_df = evaluate_model_performance("DnCNN", dncnn, val_loader, device, dncnn_ckpt)

    # 2. Evaluate SwinIR Small
    swinir = SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8)
    swinir_ckpt = checkpoints_dir / "swinir" / "best_checkpoint.pth"
    swinir_summary, swinir_df = evaluate_model_performance("SwinIR", swinir, val_loader, device, swinir_ckpt)

    # Construct Leaderboard Table
    leaderboard_df = pd.DataFrame([dncnn_summary, swinir_summary])

    # Rank Models based on Composite Score (PSNR & SSIM)
    leaderboard_df["Composite_Score"] = leaderboard_df["PSNR"] * 0.5 + leaderboard_df["SSIM"] * 50.0 - leaderboard_df["LPIPS"] * 10.0
    leaderboard_df = leaderboard_df.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)
    leaderboard_df["Rank"] = [1, 2]

    # Save Leaderboard CSV
    csv_path = metrics_dir / "stage3_model_comparison.csv"
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n[Leaderboard Saved] Metrics exported to: {csv_path}")
    print(leaderboard_df[["Rank", "Model", "PSNR", "SSIM", "LPIPS", "Inference_Time_ms", "Model_Size_MB"]])

    # Copy Best Model Checkpoint to best_model.pth
    best_model_name = leaderboard_df.iloc[0]["Model"].lower()
    best_source_ckpt = checkpoints_dir / best_model_name / "best_checkpoint.pth"
    best_dest_ckpt = checkpoints_dir / "best_model.pth"

    if best_source_ckpt.exists():
        shutil.copy(best_source_ckpt, best_dest_ckpt)
        print(f"\n[Best Model Saved] Selected {leaderboard_df.iloc[0]['Model']} as top model. Saved to: {best_dest_ckpt}")

    return leaderboard_df


if __name__ == "__main__":
    prep_dir = PROJECT_DIR / "stage2" / "preprocessed"
    run_stage3_evaluation(str(prep_dir))
