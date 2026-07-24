import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Stage3ReportGenerator:
    """Generates comprehensive technical report for Stage 3 AI MRI Enhancement."""

    @staticmethod
    def generate_report(comparison_csv_path: str, output_report_path: str):
        metrics_file = Path(comparison_csv_path)
        out_file = Path(output_report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not metrics_file.exists():
            print(f"[Error] Comparison file not found: {metrics_file}")
            return

        df = pd.read_csv(metrics_file)
        winning = df.iloc[0]
        total_models = len(df)

        # Load training histories for total time
        logs_dir = PROJECT_DIR / "stage3" / "logs"
        total_training_time = 0
        total_epochs = 0
        for hist_file in logs_dir.glob("*_history.json"):
            with open(hist_file, "r") as f:
                data = json.load(f)
            total_training_time += data.get("total_train_time_sec", 0)
            total_epochs += data.get("epochs_trained", 0)

        # Build model details table
        model_details = []
        for _, row in df.iterrows():
            model_details.append({
                "Rank": row["Rank"],
                "Model": row["Model"],
                "PSNR (dB)": f"{row['PSNR']:.4f}",
                "SSIM": f"{row['SSIM']:.4f}",
                "LPIPS": f"{row['LPIPS']:.4f}",
                "FSIM": f"{row['FSIM']:.4f}",
                "NIQE": f"{row['NIQE']:.4f}",
                "BRISQUE": f"{row['BRISQUE']:.4f}",
                "RMSE": f"{row['RMSE']:.4f}",
                "Inference (ms)": f"{row['Inference_Time_ms']:.1f}",
                "Size (MB)": f"{row['Model_Size_MB']:.2f}",
                "Params": f"{int(row['Params']):,}",
                "Score": f"{row['Composite_Score']:.4f}",
            })

        details_df = pd.DataFrame(model_details)

        report_md = f"""# MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
## Stage 3: AI-Based MRI Image Enhancement — Technical Report

---

### Executive Summary

Stage 3 implements a **research-grade deep learning framework** for Brain and Spine MRI image enhancement. The framework trains **{total_models} state-of-the-art restoration models** under identical conditions, automatically benchmarks them using **17 quantitative image quality metrics**, and selects the optimal model via a weighted composite scoring system.

**Total Training Time**: {total_training_time:.1f} seconds across {total_epochs} epochs
**Best Model**: **{winning['Model']}** with PSNR = **{winning['PSNR']:.4f} dB** and SSIM = **{winning['SSIM']:.4f}**

---

### Methodology

#### Models Benchmarked
{chr(10).join([f"{i+1}. **{m}** — ({df[df['Model']==m]['Params'].values[0]:,} params)" for i, m in enumerate(df['Model'].unique())])}

#### Training Configuration
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing with Warm Restarts + Linear Warmup (5 epochs)
- **Loss**: Hybrid Loss = 0.35×Charbonnier + 0.25×SSIM + 0.20×Perceptual(VGG) + 0.20×Edge(Sobel)
- **Max Epochs**: 300 | **Early Stopping Patience**: 20
- **Mixed Precision**: AMP enabled | **Gradient Accumulation**: 2 steps | **Gradient Clipping**: 1.0
- **Data Augmentation**: Rotation, Affine, Elastic, Gamma, Intensity, Contrast, Brightness, Noise, Flip, Cropping
- **Seed**: 42

#### Evaluation Metrics (17)
- **Full-Reference**: PSNR, SSIM, LPIPS, FSIM, VIF, UQI, MSE, RMSE
- **No-Reference**: NIQE, BRISQUE, PIQE
- **Statistical**: Entropy, Contrast, Sharpness, EdgeStrength, NoiseLevel
- **Infrastructure**: Inference Time, GPU Memory, Model Size

---

### Model Leaderboard

{details_df.to_markdown(index=False)}

---

### Winning Model Selection

- **Top Model**: **{winning['Model']}**
- **Rank**: 1 / {total_models}
- **Composite Score**: **{winning['Composite_Score']:.4f}**
- **Peak PSNR**: **{winning['PSNR']:.4f} dB**
- **Peak SSIM**: **{winning['SSIM']:.4f}**
- **Inference Speed**: **{winning['Inference_Time_ms']:.1f} ms/slice**
- **Model Size**: **{winning['Model_Size_MB']:.2f} MB** ({int(winning['Params']):,} parameters)
- **Best Checkpoint**: `stage3/checkpoints/best_model.pth`

---

### Ranking Methodology

Models are ranked using a weighted composite score:

| Metric | Direction | Weight |
|--------|-----------|--------|
| PSNR | Higher is better | 0.30 |
| SSIM | Higher is better | 0.25 |
| LPIPS | Lower is better | 0.15 |
| NIQE | Lower is better | 0.15 |
| BRISQUE | Lower is better | 0.15 |

Each metric is min-max normalized across all models to [0, 1] before weighting, ensuring fair cross-metric comparison regardless of scale.

---

### Artifacts Generated

```
stage3/
├── checkpoints/          # Best and periodic model checkpoints
├── logs/                 # Per-model training history JSON files
├── metrics/              # Model comparison CSV + per-model detailed metrics
├── reports/              # This report (Markdown + PDF)
└── visualizations/
    ├── charts/           # Loss curves, PSNR, LR schedule, radar, bar charts
    └── comparisons/      # Side-by-side MRI enhancement comparisons
```

---

*Generated automatically by Stage 3 MRI Enhancement Pipeline — MedhaDrishti Hackathon*
"""

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[Success] Report generated: {out_file}")

        # Also generate CSV summary of training runs
        summary_rows = []
        for hist_file in logs_dir.glob("*_history.json"):
            with open(hist_file, "r") as f:
                data = json.load(f)
            history = data.get("history", [])
            if history:
                summary_rows.append({
                    "Model": data.get("model_name", ""),
                    "Epochs Trained": data.get("epochs_trained", len(history)),
                    "Total Time (s)": round(data.get("total_train_time_sec", 0), 1),
                    "Best Val PSNR": data.get("best_val_psnr", 0),
                    "Best Val Loss": data.get("best_val_loss", 0),
                    "Parameters": data.get("param_count", 0),
                    "Model Size (MB)": data.get("model_size_mb", 0),
                })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(
                PROJECT_DIR / "stage3" / "metrics" / "training_summary.csv",
                index=False
            )


if __name__ == "__main__":
    csv_p = PROJECT_DIR / "stage3" / "metrics" / "stage3_model_comparison.csv"
    report_p = PROJECT_DIR / "stage3" / "reports" / "stage3_enhancement_report.md"
    Stage3ReportGenerator.generate_report(str(csv_p), str(report_p))
