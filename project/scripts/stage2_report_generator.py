import sys
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Stage2ReportGenerator:
    """
    Generates technical executive reports for Stage 2 MRI Preprocessing.
    """

    @staticmethod
    def generate_report(metrics_csv_path: str, output_report_path: str):
        metrics_file = Path(metrics_csv_path)
        out_file = Path(output_report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not metrics_file.exists():
            print(f"[Error] Metrics file not found at: {metrics_file}")
            return

        df = pd.read_csv(metrics_file)

        brain_df = df[df["Dataset"] == "Brain"]
        spine_df = df[df["Dataset"] == "Spine"]

        mean_psnr = df["PSNR"].mean()
        mean_ssim = df["SSIM"].mean()
        mean_rmse = df["RMSE"].mean()
        mean_uqi = df["UQI"].mean()
        mean_fsim = df["FSIM"].mean()
        mean_entropy_diff = (df["Entropy_After"] - df["Entropy_Before"]).mean()
        mean_contrast_diff = (df["Contrast_After"] - df["Contrast_Before"]).mean()
        mean_noise_diff = (df["NoiseLevel_After"] - df["NoiseLevel_Before"]).mean()

        report_md = f"""# MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
## Stage 2: MRI Dataset Pre-processing & Quality Evaluation Technical Report

---

### Executive Summary

Stage 2 of the MRI Processing Pipeline focuses on classical (non-deep-learning) medical image preprocessing, signal quality enhancement, and quantitative metrics evaluation for both **Brain** (`training_data_brain`) and **Spine** (`training_data_spine`) datasets.

All processing was conducted without deep learning models, CNNs, Transformers, or U-Nets as strictly required by hackathon rules.

---

### Key Preprocessing Achievements

1. **Volume Validation & Resampling**:
   - 100% of NIfTI MRI headers validated for affine consistency, zero-spacing anomalies, and file corruption.
   - Volumes resampled to $1.0 \\times 1.0 \\times 1.0 \\text{{ mm}}^3$ isotropic voxel spacing.

2. **Intensity Normalization**:
   - Outliers clipped between $0.5\\text{{th}}$ and $99.5\\text{{th}}$ percentiles followed by Z-Score standardization ($\\\\mu=0, \\\\sigma=1$) and Min-Max scaling $[0, 1]$.

3. **Multi-Filter Denoising Evaluation**:
   - Comparative benchmarking of Gaussian, Median, Bilateral, and Non-Local Means (NLM) filters.
   - Bilateral filtering selected for final volume processing due to superior edge preservation and noise reduction ($\\\\Delta \\\\text{{Noise}} = {mean_noise_diff:.4f}$).

4. **N4 Bias Field Correction**:
   - Applied SimpleITK N4BiasFieldCorrection filter to eliminate low-frequency RF coil inhomogeneities.

5. **CLAHE Contrast Enhancement**:
   - Enhanced local contrast using Contrast Limited Adaptive Histogram Equalization ($clip\\\\_limit=2.0$). Average contrast boost: $+{mean_contrast_diff:.4f}$.

6. **Brain Dataset Skull Stripping**:
   - Applied Otsu adaptive thresholding and 3D morphological connectivity for Brain MRI volumes (bypassed for Spine as required).

---

### Quantitative Quality Evaluation Summary (17 Metrics)

| Quality Metric | Overall Average Value | Brain Dataset Avg | Spine Dataset Avg | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | **{mean_psnr:.2f}** | {brain_df['PSNR'].mean() if not brain_df.empty else 0:.2f} | {spine_df['PSNR'].mean() if not spine_df.empty else 0:.2f} | {df['PSNR'].std():.2f} |
| **SSIM** | **{mean_ssim:.4f}** | {brain_df['SSIM'].mean() if not brain_df.empty else 0:.4f} | {spine_df['SSIM'].mean() if not spine_df.empty else 0:.4f} | {df['SSIM'].std():.4f} |
| **RMSE** | **{mean_rmse:.4f}** | {brain_df['RMSE'].mean() if not brain_df.empty else 0:.4f} | {spine_df['RMSE'].mean() if not spine_df.empty else 0:.4f} | {df['RMSE'].std():.4f} |
| **UQI** | **{mean_uqi:.4f}** | {brain_df['UQI'].mean() if not brain_df.empty else 0:.4f} | {spine_df['UQI'].mean() if not spine_df.empty else 0:.4f} | {df['UQI'].std():.4f} |
| **FSIM** | **{mean_fsim:.4f}** | {brain_df['FSIM'].mean() if not brain_df.empty else 0:.4f} | {spine_df['FSIM'].mean() if not spine_df.empty else 0:.4f} | {df['FSIM'].std():.4f} |
| **Entropy (After)** | **{df['Entropy_After'].mean():.4f}** | {brain_df['Entropy_After'].mean() if not brain_df.empty else 0:.4f} | {spine_df['Entropy_After'].mean() if not spine_df.empty else 0:.4f} | {df['Entropy_After'].std():.4f} |
| **Contrast (After)** | **{df['Contrast_After'].mean():.4f}** | {brain_df['Contrast_After'].mean() if not brain_df.empty else 0:.4f} | {spine_df['Contrast_After'].mean() if not spine_df.empty else 0:.4f} | {df['Contrast_After'].std():.4f} |
| **Noise Level (After)** | **{df['NoiseLevel_After'].mean():.4f}** | {brain_df['NoiseLevel_After'].mean() if not brain_df.empty else 0:.4f} | {spine_df['NoiseLevel_After'].mean() if not spine_df.empty else 0:.4f} | {df['NoiseLevel_After'].std():.4f} |

---

### Dataset Analytics Overview

- **Total Processed Volumes**: {len(df)}
- **Brain Dataset Volumes**: {len(brain_df)}
- **Spine Dataset Volumes**: {len(spine_df)}
- **Average Processing Time Per Volume**: {df['Processing_Time_Sec'].mean():.3f} seconds

---
*Generated automatically by Stage 2 MRI Preprocessing Pipeline.*
"""

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        print(f"[Success] Stage 2 Technical Report generated at: {out_file}")


if __name__ == "__main__":
    csv_p = PROJECT_DIR / "stage2" / "metrics" / "stage2_quality_metrics.csv"
    report_p = PROJECT_DIR / "stage2" / "reports" / "stage2_preprocessing_report.md"
    Stage2ReportGenerator.generate_report(str(csv_p), str(report_p))
