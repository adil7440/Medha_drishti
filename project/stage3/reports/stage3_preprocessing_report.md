# MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
## Stage 3: AI-Based MRI Image Enhancement Technical Report

---

### Executive Summary

Stage 3 focuses on deep learning AI-based enhancement for Brain and Spine MRI images. A lightweight benchmarking framework was constructed comparing **SwinIR Small** (Primary Model) against **DnCNN** (Baseline Model) under identical training and evaluation conditions.

Both models were trained using PyTorch GPU acceleration with Mixed Precision (AMP), AdamW optimizer ($	ext{lr}=1	ext{e-}4$), Cosine Annealing learning rate scheduler, and Early Stopping ($	ext{patience}=5$).

---

### Model Architecture Comparison

1. **SwinIR Small (Primary Model)**:
   - **Type**: Lightweight Swin Transformer for Image Restoration.
   - **Feature Extraction**: 3 Residual Swin Transformer Blocks (RSTB) with Window-based Multi-Head Self-Attention (W-MSA & SW-MSA).
   - **Parameter Count**: 203,209 parameters.

2. **DnCNN (Baseline Model)**:
   - **Type**: Deep Residual Convolutional Neural Network (17 layers).
   - **Feature Extraction**: Cascaded 3x3 Conv + BatchNorm + ReLU layers with residual noise learning.
   - **Parameter Count**: 556,032 parameters.

---

### Quantitative Model Leaderboard & Evaluation (14 Metrics)

|   Rank | Model   |   PSNR |   SSIM |   LPIPS |      MSE |   RMSE |    UQI |   FSIM |   Inference_Time_ms |   Model_Size_MB |
|-------:|:--------|-------:|-------:|--------:|---------:|-------:|-------:|-------:|--------------------:|----------------:|
|      1 | DnCNN   |  21.59 | 0.3922 |  0.0561 | 0.006996 | 0.0835 | 0.8943 | 0.9193 |               45.78 |            2.12 |
|      2 | SwinIR  |  18.46 | 0.3587 |  0.1082 | 0.014578 | 0.1201 | 0.8311 | 0.7767 |               80.45 |            0.78 |

---

### Winning Model Selection

- **Top Model**: **DnCNN**
- **Peak PSNR**: **21.59 dB**
- **Peak SSIM**: **0.3922**
- **Inference Speed**: **45.78 ms/slice**
- **Best Weight Checkpoint**: `stage3/checkpoints/best_model.pth`

---
*Generated automatically by Stage 3 MRI Enhancement Pipeline.*
