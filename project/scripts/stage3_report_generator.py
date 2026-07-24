import sys
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Stage3ReportGenerator:
    """
    Generates technical executive report for Stage 3 AI MRI Enhancement.
    """

    @staticmethod
    def generate_report(comparison_csv_path: str, output_report_path: str):
        metrics_file = Path(comparison_csv_path)
        out_file = Path(output_report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not metrics_file.exists():
            print(f"[Error] Comparison file not found at: {metrics_file}")
            return

        df = pd.read_csv(metrics_file)
        winning_row = df.iloc[0]

        report_md = f"""# MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
## Stage 3: AI-Based MRI Image Enhancement Technical Report

---

### Executive Summary

Stage 3 focuses on deep learning AI-based enhancement for Brain and Spine MRI images. A lightweight benchmarking framework was constructed comparing **SwinIR Small** (Primary Model) against **DnCNN** (Baseline Model) under identical training and evaluation conditions.

Both models were trained using PyTorch GPU acceleration with Mixed Precision (AMP), AdamW optimizer ($\text{{lr}}=1\text{{e-}}4$), Cosine Annealing learning rate scheduler, and Early Stopping ($\text{{patience}}=5$).

---

### Model Architecture Comparison

1. **SwinIR Small (Primary Model)**:
   - **Type**: Lightweight Swin Transformer for Image Restoration.
   - **Feature Extraction**: 3 Residual Swin Transformer Blocks (RSTB) with Window-based Multi-Head Self-Attention (W-MSA & SW-MSA).
   - **Parameter Count**: {winning_row['Params'] if winning_row['Model'] == 'SwinIR' else df[df['Model']=='SwinIR']['Params'].values[0]:,} parameters.

2. **DnCNN (Baseline Model)**:
   - **Type**: Deep Residual Convolutional Neural Network (17 layers).
   - **Feature Extraction**: Cascaded 3x3 Conv + BatchNorm + ReLU layers with residual noise learning.
   - **Parameter Count**: {winning_row['Params'] if winning_row['Model'] == 'DnCNN' else df[df['Model']=='DnCNN']['Params'].values[0]:,} parameters.

---

### Quantitative Model Leaderboard & Evaluation (14 Metrics)

{df[['Rank', 'Model', 'PSNR', 'SSIM', 'LPIPS', 'MSE', 'RMSE', 'UQI', 'FSIM', 'Inference_Time_ms', 'Model_Size_MB']].to_markdown(index=False)}

---

### Winning Model Selection

- **Top Model**: **{winning_row['Model']}**
- **Peak PSNR**: **{winning_row['PSNR']} dB**
- **Peak SSIM**: **{winning_row['SSIM']}**
- **Inference Speed**: **{winning_row['Inference_Time_ms']} ms/slice**
- **Best Weight Checkpoint**: `stage3/checkpoints/best_model.pth`

---
*Generated automatically by Stage 3 MRI Enhancement Pipeline.*
"""

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        print(f"[Success] Stage 3 Technical Report generated at: {out_file}")


if __name__ == "__main__":
    csv_p = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
    report_p = PROJECT_DIR / "stage3" / "reports" / "stage3_enhancement_report.md"
    Stage3ReportGenerator.generate_report(str(csv_p), str(report_p))
