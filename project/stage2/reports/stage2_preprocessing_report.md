# MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
## Stage 2: MRI Dataset Pre-processing & Quality Evaluation Technical Report

---

### Executive Summary

Stage 2 of the MRI Processing Pipeline focuses on classical (non-deep-learning) medical image preprocessing, signal quality enhancement, and quantitative metrics evaluation for both **Brain** (`training_data_brain`) and **Spine** (`training_data_spine`) datasets.

All processing was conducted without deep learning models, CNNs, Transformers, or U-Nets as strictly required by hackathon rules.

---

### Key Preprocessing Achievements

1. **Volume Validation & Resampling**:
   - 100% of NIfTI MRI headers validated for affine consistency, zero-spacing anomalies, and file corruption.
   - Volumes resampled to $1.0 \times 1.0 \times 1.0 \text{ mm}^3$ isotropic voxel spacing.

2. **Intensity Normalization**:
   - Outliers clipped between $0.5\text{th}$ and $99.5\text{th}$ percentiles followed by Z-Score standardization ($\\mu=0, \\sigma=1$) and Min-Max scaling $[0, 1]$.

3. **Multi-Filter Denoising Evaluation**:
   - Comparative benchmarking of Gaussian, Median, Bilateral, and Non-Local Means (NLM) filters.
   - Bilateral filtering selected for final volume processing due to superior edge preservation and noise reduction ($\\Delta \\text{Noise} = -0.0052$).

4. **N4 Bias Field Correction**:
   - Applied SimpleITK N4BiasFieldCorrection filter to eliminate low-frequency RF coil inhomogeneities.

5. **CLAHE Contrast Enhancement**:
   - Enhanced local contrast using Contrast Limited Adaptive Histogram Equalization ($clip\\_limit=2.0$). Average contrast boost: $+0.0811$.

6. **Brain Dataset Skull Stripping**:
   - Applied Otsu adaptive thresholding and 3D morphological connectivity for Brain MRI volumes (bypassed for Spine as required).

---

### Quantitative Quality Evaluation Summary (17 Metrics)

| Quality Metric | Overall Average Value | Brain Dataset Avg | Spine Dataset Avg | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | **18.41** | 22.16 | 15.99 | 3.73 |
| **SSIM** | **0.7842** | 0.9069 | 0.7049 | 0.1236 |
| **RMSE** | **0.1306** | 0.0817 | 0.1622 | 0.0502 |
| **UQI** | **0.8308** | 0.8920 | 0.7913 | 0.1059 |
| **FSIM** | **0.8196** | 0.7977 | 0.8338 | 0.0605 |
| **Entropy (After)** | **5.2754** | 3.1506 | 6.6463 | 2.0272 |
| **Contrast (After)** | **0.2436** | 0.2027 | 0.2699 | 0.0435 |
| **Noise Level (After)** | **0.1118** | 0.2443 | 0.0264 | 0.1146 |

---

### Dataset Analytics Overview

- **Total Processed Volumes**: 306
- **Brain Dataset Volumes**: 120
- **Spine Dataset Volumes**: 186
- **Average Processing Time Per Volume**: 1.769 seconds

---
*Generated automatically by Stage 2 MRI Preprocessing Pipeline.*
