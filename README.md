# Yugma — Multi-Stage MRI Medical Image Processing Pipeline

> **MedhaDrishti National-Level AI Hackathon | Yugma TechFest 2.0**

A comprehensive 4-stage medical image processing pipeline for **Brain MRI** (BraTS 2020) and **Spine MRI** (pathological/normal), featuring classical preprocessing, AI-based enhancement with 6 deep learning models, segmentation, and an interactive Streamlit dashboard with 29 pages.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Stage 1 — Dataset Exploration & Analysis](#stage-1--dataset-exploration--analysis)
- [Stage 2 — Preprocessing Pipeline](#stage-2--preprocessing-pipeline)
- [Stage 3 — AI-Based Enhancement](#stage-3--ai-based-enhancement)
  - [Models](#models)
  - [Training Configuration](#training-configuration)
  - [Loss Function — HybridLoss](#loss-function--hybridloss)
  - [Evaluation & Ranking](#evaluation--ranking)
  - [Training Results](#training-results)
- [Stage 4 — Segmentation & Inference](#stage-4--segmentation--inference)
- [Interactive Dashboard](#interactive-dashboard)
- [Spine Pipeline](#spine-pipeline)
- [PDF Report Generation](#pdf-report-generation)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Quality Metrics Reference](#quality-metrics-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YUGMA PIPELINE                               │
├──────────┬──────────┬──────────────┬────────────────────────────────┤
│ STAGE 1  │ STAGE 2  │   STAGE 3    │          STAGE 4               │
│ Dataset  │ Preproc- │ AI Enhancement│  Segmentation &               │
│ Explorer │ essing   │  (6 Models)  │  Clinical Reports              │
├──────────┼──────────┼──────────────┼────────────────────────────────┤
│ • Scan   │ • Valid. │ • DnCNN      │ • Inference                    │
│ • Props  │ • Resamp │ • SwinIR-S   │ • Mask Generation              │
│ • Stats  │ • Normal │ • SwinIR-L   │ • Report Generation            │
│ • Visual │ • Denoise│ • Restormer  │                                │
│ • Report │ • N4 Bias│ • MIRNet-v2  │                                │
│ • Notebk │ • CLAHE  │ • NAFNet     │                                │
│          │ • Skull  │ • Ranking    │                                │
│          │ • Augment│ • Charts     │                                │
├──────────┴──────────┴──────────────┴────────────────────────────────┤
│              Streamlit Dashboard (29 Pages)                          │
│     Patient Explorer | Viewer | Metrics | Leaderboard | Reports     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

```
Raw NIfTI Volumes → Stage 1 (Analysis CSVs) → Stage 2 (Preprocessed NPZ)
  → Stage 3 (AI-Enhanced NPZ) → Stage 4 (Segmentation Masks + Clinical PDF)
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.13 |
| **Deep Learning** | PyTorch ≥2.0, torchvision ≥0.15 |
| **Medical Image I/O** | nibabel ≥4.0, SimpleITK ≥2.2 |
| **Image Processing** | OpenCV ≥4.6, scikit-image ≥0.19, scipy ≥1.8 |
| **Data & Numerics** | NumPy ≥1.21, pandas ≥1.4 |
| **Visualization** | matplotlib ≥3.5, Plotly ≥5.10 |
| **Dashboard** | Streamlit ≥1.20 |
| **Report Generation** | ReportLab ≥3.6 |
| **Config** | PyYAML ≥6.0 |
| **Utilities** | tqdm ≥4.64, tabulate ≥0.9 |

---

## Directory Structure

```
yugma/
├── project/                          # Main source code
│   ├── main.py                       # Stage 1 brain orchestrator
│   ├── spine_main.py                 # Stage 1 spine orchestrator
│   ├── run_stage3.py                 # Stage 3 AI enhancement orchestrator
│   ├── run_dashboard.py              # Streamlit dashboard launcher
│   ├── generate_stage2_pdf.py        # Stage 2 preprocessing PDF (1021 lines)
│   ├── generate_hackathon_pdf.py     # Hackathon evaluation PDF (1275 lines)
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── models/                       # 6 PyTorch model architectures
│   │   ├── __init__.py
│   │   ├── dncnn.py                  # DnCNN with SE blocks + U-Net
│   │   ├── swinir.py                 # SwinIR Small (lightweight)
│   │   ├── swinir_large.py           # SwinIR Large (high quality)
│   │   ├── restormer.py              # Restormer (linear attention)
│   │   ├── mirnet_v2.py              # MIRNet-v2 (CBAM + multi-scale)
│   │   └── nafnet.py                 # NAFNet (activation-free)
│   │
│   ├── configs/
│   │   └── stage3_config.yaml        # Master config (219 lines)
│   │
│   ├── notebooks/
│   │   └── Stage1_Analysis.ipynb     # Interactive Jupyter notebook
│   │
│   ├── dashboard/                    # Streamlit multi-page app
│   │   ├── app.py                    # Main entry point
│   │   ├── components/               # Reusable UI components
│   │   │   ├── metrics_cards.py
│   │   │   ├── image_comparison.py
│   │   │   └── plotly_charts.py
│   │   └── pages/                    # 29 dashboard pages
│   │       ├── 1_Home.py
│   │       ├── 2_Patient_Explorer/
│   │       ├── ...
│   │       └── 29_Downloads/
│   │
│   ├── stage3/                       # Stage 3 outputs
│   │   ├── checkpoints/              # Model weights
│   │   ├── logs/                     # Training histories (JSON)
│   │   ├── metrics/                  # Evaluation CSVs
│   │   ├── reports/                  # Markdown reports
│   │   └── visualizations/           # Charts + comparisons
│   │
│   ├── stage4/                       # Stage 4 outputs
│   │   ├── uploads/                  # Input NIfTI volumes
│   │   ├── segmentations/            # Generated masks
│   │   └── reports/                  # Clinical PDFs
│   │
│   └── figures/                      # Generated visualizations
│       ├── stage2/                   # Preprocessing figures
│       └── spine/                    # Spine-specific figures
│
├── scripts/                          # Reusable script modules
│   ├── dataset_loader.py             # Brain NIfTI discovery
│   ├── image_properties.py           # MRI property extraction
│   ├── statistics.py                 # Statistics computation
│   ├── visualization.py              # Matplotlib figures
│   ├── report_generator.py           # Markdown/PDF reports
│   ├── generate_notebook.py          # Jupyter notebook creation
│   ├── stage2_pipeline.py            # 8-step preprocessing
│   ├── quality_evaluator.py          # 17 quality metrics
│   ├── mri_validator.py              # NIfTI validation
│   ├── resampler.py                  # Voxel resampling
│   ├── intensity_normalizer.py       # Percentile normalization
│   ├── noise_remover.py              # Denoising filters
│   ├── bias_field_corrector.py       # N4 bias correction
│   ├── contrast_enhancer.py          # CLAHE enhancement
│   ├── skull_stripper.py             # Otsu skull stripping
│   ├── data_augmentor.py             # Data augmentation
│   ├── loss_functions.py             # HybridLoss for Stage 3
│   ├── train_stage3.py               # Model training loop
│   ├── evaluate_stage3.py            # Model evaluation + leaderboard
│   ├── stage3_dataset.py             # Training dataset loader
│   ├── stage3_report_generator.py    # Stage 3 report
│   ├── stage3_visualization.py       # Comparison plots
│   ├── training_visualization.py     # Training curve charts
│   ├── stage4_inference.py           # Segmentation inference
│   ├── stage4_report_generator.py    # Stage 4 report
│   ├── spine_dataset_loader.py       # Spine NIfTI discovery
│   ├── spine_image_properties.py     # Spine property extraction
│   ├── spine_statistics.py           # Spine statistics
│   ├── spine_visualization.py        # Spine figures
│   ├── spine_report_generator.py     # Spine report
│   ├── spine_split.py                # Train/test splitter
│   └── stage2_report_generator.py    # Stage 2 report
│
├── training_data_brain/              # BraTS 2020 brain MRI (NIfTI)
├── training_data_spine/              # Spine MRI (NIfTI, 20 patients)
├── test_brain/                       # Test brain dataset (BRP1-BRP10)
├── test_spine/                       # Test spine dataset (SP11-SP23)
└── validation_brain/                 # BraTS 2020 validation data
```

---

## Stage 1 — Dataset Exploration & Analysis

**Entry Points:** `python main.py` (Brain), `python spine_main.py` (Spine)

### Pipeline Steps

1. **Directory Setup** — Creates `analysis/`, `figures/`, `reports/`, `notebooks/` output directories
2. **Dataset Discovery** — `BrainDatasetLoader` / `SpineDatasetLoader` scans NIfTI directories, identifies patients and modalities
3. **Parallel Property Extraction** — `ProcessPoolExecutor` with `os.cpu_count()-1` workers extracts per-volume image properties:
   - Dimensions (Width, Height, Depth)
   - Voxel Spacing (X, Y, Z)
   - Mean Intensity, Contrast (RMS), Entropy (Shannon), SNR
4. **Statistics & Quality Checks** — `DatasetStatisticsCalculator` computes per-modality statistics and flags quality warnings
5. **Visualization** — `DatasetVisualizer` generates modality comparison figures, heatmaps, boxplots
6. **Report Generation** — `ReportGenerator` produces Markdown + PDF reports
7. **Notebook Creation** — Auto-generated Jupyter notebook for interactive exploration

### Datasets

| Dataset | Patients | Modalities | Format |
|---------|----------|------------|--------|
| Brain (Training) | BraTS 2020 | T1, T1CE, T2, FLAIR, SEG | `.nii` |
| Brain (Test) | BRP1–BRP10 | T1, T1CE, T2, FLAIR | `.nii` |
| Spine (Training) | 10 (5 Normal + 5 Pathological) | T1W, T2W, STIR, SPAIR, T1W_GADO, Survey | `.nii.gz` |
| Spine (Test) | 10 (5 Normal + 5 Pathological) | T1W, T2W, STIR, SPAIR, T1W_GADO, Survey | `.nii.gz` |

### 7 Image Properties Extracted

| Property | Method |
|----------|--------|
| **Contrast** | RMS contrast (Std/Mean of foreground) |
| **Complexity** | Shannon Entropy (256-bin histogram) |
| **Sharpness** | 3D Laplacian Variance |
| **Edge Strength** | Sobel Gradient Magnitude |
| **Noise Level** | Background MAD / 0.6745 |
| **Mean Intensity** | Average foreground voxel |
| **Standard Deviation** | Foreground voxel standard deviation |

---

## Stage 2 — Preprocessing Pipeline

**Entry Point:** `scripts/stage2_pipeline.py`

### 8-Step Pipeline

| Step | Process | Key Parameters |
|------|---------|---------------|
| 1 | **MRI Validation** | NIfTI format check, modality verification |
| 2 | **Voxel Resampling** | Target isotropic spacing |
| 3 | **Intensity Normalization** | Percentile clipping + min-max scaling |
| 4 | **Denoising** | Gaussian, Median, Bilateral, NLM filters |
| 5 | **N4 Bias Field Correction** | Iterations: [30, 20, 10] |
| 6 | **CLAHE Enhancement** | clip_limit=2.0, tile_grid_size=(8,8) |
| 7 | **Skull Stripping** | Otsu threshold × 0.3 + morphological ops |
| 8 | **Data Augmentation** | Rotation=10°, Flip, Gamma=1.2, Gaussian Noise (std=0.01) |

### Quality Evaluation — 17 Metrics

| # | Metric | Direction | Description |
|---|--------|-----------|-------------|
| 1 | **PSNR** | ↑ | Peak Signal-to-Noise Ratio |
| 2 | **SSIM** | ↑ | Structural Similarity Index |
| 3 | **MSE** | ↓ | Mean Squared Error |
| 4 | **RMSE** | ↓ | Root Mean Squared Error |
| 5 | **UQI** | ↑ | Universal Quality Index |
| 6 | **FSIM** | ↑ | Feature Similarity Index |
| 7 | **GMSD** | ↓ | Gradient Magnitude Similarity Deviation |
| 8 | **VIF** | ↑ | Visual Information Fidelity |
| 9 | **BRISQUE** | ↓ | Blind/Referenceless Image Spatial Quality |
| 10 | **NIQE** | ↓ | Natural Image Quality Evaluator |
| 11 | **PIQE** | ↓ | Perception based Image Quality Evaluator |
| 12 | **LPIPS** | ↓ | Learned Perceptual Image Patch Similarity |
| 13 | **Entropy** | ↑ | Information content (256-bin histogram) |
| 14 | **Contrast** | ↑ | RMS contrast |
| 15 | **Sharpness** | ↑ | Laplacian variance |
| 16 | **Edge Strength** | ↑ | Sobel gradient magnitude |
| 17 | **Noise Level** | ↓ | Background MAD estimator |

---

## Stage 3 — AI-Based Enhancement

**Entry Point:** `python run_stage3.py`

### 5-Step Pipeline

1. **Train** all enabled models
2. **Evaluate & Rank** models → leaderboard CSV
3. **Generate Training Charts** — loss curves, PSNR curves, LR schedules
4. **Generate Comparison Visualizations** — before/after, edge maps
5. **Generate Report** — comprehensive Markdown report

### Models

#### 1. DnCNN (Enabled)

| Property | Value |
|----------|-------|
| **Architecture** | U-Net encoder-decoder with Squeeze-and-Excitation blocks |
| **Encoder** | 3 levels: 64 → 128 → 256 channels |
| **Decoder** | ConvTranspose2d upsampling + skip concatenation |
| **Key Mechanism** | Residual learning — predicts noise, subtracts from input |
| **Parameters** | ~4.2M |
| **Model Size** | ~16.1 MB |

```
Input → DoubleConv(1→64) → MaxPool → DoubleConv(64→128) → MaxPool → DoubleConv(128→256)
  → ConvTranspose → Concat → DoubleConv(128→64) → ConvTranspose → Concat → DoubleConv(64→64)
  → Conv3x3 → Residual Subtraction → Output
```

Each `DoubleConv` block: `(Conv3x3 → BN → ReLU) × 2 + SEBlock(reduction=16)`

#### 2. SwinIR Small

| Property | Value |
|----------|-------|
| **Architecture** | Swin Transformer for Image Restoration |
| **Embed Dim** | 48 |
| **Heads** | 4 |
| **Window Size** | 8 |
| **RSTB Blocks** | 3 (depth=2 each) |
| **Key Mechanism** | Shifted window attention with relative position bias |

```
Input → Conv3x3(1→48) → [RSTB₁ → RSTB₂ → RSTB₃] → Conv3x3 → + Global Residual → Conv3x3(48→1) → + Residual → Output
```

#### 3. SwinIR Large

| Property | Value |
|----------|-------|
| **Embed Dim** | 180 |
| **Heads** | 6 |
| **Window Size** | 8 |
| **RSTB Blocks** | 4 (depth=[6,6,6,6]) |
| **MLP Ratio** | 4.0× |

Deeper and wider variant of SwinIR Small for maximum restoration quality.

#### 4. Restormer

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-decoder Transformer |
| **Dim** | 48 |
| **Blocks** | [4, 6, 6, 8] |
| **Heads** | [1, 2, 4, 8] |
| **Key Mechanism** | **Linear Attention** O(n) + GDFN gated feed-forward |

```
Input → Encoder (2 levels, strided conv) → Bottleneck (6 TransformerBlocks)
  → Decoder (2 levels, ConvTranspose + skip) → Refinement (4 TransformerBlocks) → + Residual → Output
```

- **Linear Attention:** `ELU(k)+1` / `ELU(v)+1` — avoids softmax O(n²) bottleneck
- **GDFN:** Gated-Dconv FFN — `GELU(x₁) × x₂` gating mechanism

#### 5. MIRNet-v2

| Property | Value |
|----------|-------|
| **Architecture** | Multi-scale Residual Network |
| **Features** | [32, 64, 128] |
| **Blocks/Group** | 2 |
| **Key Mechanism** | CBAM attention + multi-scale conv (3×3, 5×5, 7×7) |

```
Input → Encoder (ResBlock + Downsample) × 3
  → MultiScaleBlock (3 parallel conv branches → concat → fuse)
  → Decoder (Upsample + skip) × 3 → + Residual → Output
```

- **CBAM:** Channel Attention (dual pooling → FC → sigmoid) + Spatial Attention (cat → Conv7×7 → sigmoid)

#### 6. NAFNet

| Property | Value |
|----------|-------|
| **Architecture** | Nonlinear Activation Free Network |
| **Width** | 32 |
| **Encoder Blocks** | [2, 2, 4, 8] |
| **Decoder Blocks** | [2, 2, 2, 2] |
| **Key Mechanism** | SimpleGate (half-channel multiply, no activation function) |

```
Input → Encoder (4 levels: 32→64→128→256→512)
  → Middle (1 NAFBlock) → Decoder (4 levels, PixelShuffle) → + Residual → Output
```

- **SimpleGate:** Splits channels in half, element-wise multiply — replaces ReLU/GELU
- **Learnable Scaling:** `beta`, `gamma` initialized to 0 for training stability

---

### Training Configuration

**Config File:** `project/configs/stage3_config.yaml`

| Parameter | Value |
|-----------|-------|
| **Seed** | 42 |
| **Target Size** | 256×256 |
| **Train/Val Split** | 80/20 |
| **Batch Size** | 4 (CUDA) / 2 (CPU) |
| **Max Epochs** | 20 |
| **Early Stopping** | patience=20, min_delta=1e-6 |
| **Mixed Precision** | Enabled |
| **Gradient Accumulation** | 2 steps |
| **Gradient Clipping** | max_norm=1.0 |
| **Checkpoint Metric** | val_psnr |

#### Optimizer

| Parameter | Value |
|-----------|-------|
| **Algorithm** | AdamW |
| **Learning Rate** | 2×10⁻⁴ |
| **Weight Decay** | 1×10⁻⁴ |
| **Betas** | [0.9, 0.999] |

#### Scheduler

| Parameter | Value |
|-----------|-------|
| **Algorithm** | CosineAnnealingWarmRestarts |
| **T_0** | 50 |
| **T_mult** | 2 |
| **eta_min** | 1×10⁻⁷ |
| **Warmup Epochs** | 5 |

---

### Loss Function — HybridLoss

A weighted combination of 4 loss components:

| Component | Weight | Details |
|-----------|--------|---------|
| **Charbonnier Loss** | 0.10 | Smooth L1 variant, eps=1e-6 |
| **SSIM Loss** | 0.80 | Window size=11, structural similarity |
| **Perceptual Loss** | 0.05 | VGG-19 features: relu1_2, relu2_2, relu3_3, relu4_3 (equal weight) |
| **Edge Loss** | 0.05 | Sobel gradient magnitude difference |

```
L_total = 0.10 × L_charbonnier + 0.80 × L_ssim + 0.05 × L_perceptual + 0.05 × L_edge
```

---

### Evaluation & Ranking

#### 16 Quality Metrics + 3 Infrastructure Metrics

**Quality:** PSNR, SSIM, LPIPS, FSIM, VIF, UQI, MSE, RMSE, NIQE, BRISQUE, PIQE, Entropy, Contrast, Sharpness, EdgeStrength, NoiseLevel

**Infrastructure:** Inference_Time_ms, GPU_Memory_MB, Model_Size_MB

#### Weighted Composite Scoring

| Metric | Direction | Weight |
|--------|-----------|--------|
| **PSNR** | Higher is better | 0.30 |
| **SSIM** | Higher is better | 0.25 |
| **LPIPS** | Lower is better | 0.15 |
| **NIQE** | Lower is better | 0.15 |
| **BRISQUE** | Lower is better | 0.15 |

```
Score = 0.30 × norm(PSNR) + 0.25 × norm(SSIM) + 0.15 × (1-norm(LPIPS)) + 0.15 × (1-norm(NIQE)) + 0.15 × (1-norm(BRISQUE))
```

---

### Training Results

#### DnCNN (20 epochs, best at epoch 17)

| Metric | Value |
|--------|-------|
| **Best Validation PSNR** | 23.92 dB |
| **Best Validation Loss** | 0.1059 |
| **Parameters** | 4,219,873 |
| **Model Size** | 16.1 MB |
| **Total Training Time** | ~2.8 hours (10,112 sec) |
| **Avg Time/Epoch** | ~506 sec |

| Epoch | Train Loss | Val Loss | Val PSNR | LR |
|-------|-----------|----------|----------|-----|
| 1 | 0.4136 | 0.4520 | 18.51 dB | 4.0×10⁻⁵ |
| 5 | 0.2870 | 0.2885 | 21.43 dB | 2.0×10⁻⁴ |
| 11 | 0.2601 | 0.1037 | 23.16 dB | 1.93×10⁻⁴ |
| **17** | **0.2329** | **0.1059** | **23.92 dB** | 1.73×10⁻⁴ |
| 20 | 0.2423 | 0.1189 | 23.90 dB | 1.59×10⁻⁴ |

#### SwinIR Small (5 epochs, best at epoch 5)

| Metric | Value |
|--------|-------|
| **Best Validation PSNR** | 21.14 dB |
| **Best Validation Loss** | 0.1362 |
| **Total Training Time** | ~47 sec |

| Epoch | Train Loss | Val Loss | Val PSNR |
|-------|-----------|----------|----------|
| 1 | 0.3278 | 0.2788 | 14.41 dB |
| 5 | 0.1366 | 0.1362 | 21.14 dB |

---

## Stage 4 — Segmentation & Inference

**Output Directory:** `project/stage4/`

### Pipeline

1. **Upload** — Accepts NIfTI volumes via dashboard upload station
2. **Preprocessing** — Full Stage 2 pipeline
3. **AI Enhancement** — Selected model from Stage 3
4. **Segmentation** — Generates binary masks (`.nii_mask.nii.gz`)
5. **Clinical Measurements** — Volume (mm³), Max Area (mm²), Affected Tissue %
6. **Report Generation** — Clinical PDF report with diagnosis, measurements, confidence analysis

### Generated Artifacts

- `uploads/` — Input NIfTI volumes (brain + spine test cases)
- `segmentations/` — Generated segmentation masks
- `reports/` — Clinical PDF reports (final_report.pdf, patient-specific)
- `annotated_images/` — Visualization snapshots

---

## Interactive Dashboard

**Entry Point:** `python run_dashboard.py`

A **29-page Streamlit application** providing interactive exploration of the entire pipeline.

### Pages Overview

| Page | Description |
|------|-------------|
| **1. Home** | Project overview, dataset stats (369 Brain + 10 Spine patients) |
| **2. Patient Explorer** | Filter by Dataset → Patient → Modality, view metadata + quality JSON |
| **3. Before vs After** | Side-by-side original vs preprocessed with difference heatmap |
| **4. Pipeline View** | 6-stage grid: Original → Norm → Denoise → N4 → CLAHE → Final |
| **5. Slice Viewer** | Z-axis multi-slice navigation with slider |
| **6. Quality Metrics** | Full 17-metric breakdown with metric cards |
| **7. Graphs** | Radar chart, boxplots, scatter plots, intensity histograms |
| **8. Dataset Analytics** | Volume counts, processing time, summary statistics |
| **9. Augmentation Viewer** | 5-panel augmentation chain visualization |
| **10. Reports** | Download center for Stage 2 reports and metrics |
| **11. AI Enhancement Viewer** | Load 6 models, run inference, show visual progression + metrics |
| **12. Model Comparison** | Head-to-head PSNR/SSIM/LPIPS/NIQE/BRISQUE bar charts + radar |
| **13. Training Monitor** | Loss curves, PSNR curves, LR schedule, GPU memory, timing |
| **14. Metrics Dashboard** | Per-model metric cards + cross-model grouped bar comparisons |
| **15. Leaderboard** | Weighted composite scoring, Gold/Silver/Bronze podium |
| **16. Patient Viewer** | Browse all patients, display NPZ arrays, per-case quality |
| **17. Difference Heatmaps** | Pixel-level difference analysis, Canny edge comparison |
| **18. Inference Demo** | Live real-time AI inference with timing + full metrics |
| **19. Upload MRI** | Upload NIfTI/NPZ/DICOM, run full pipeline (Stages 2→3→4) |
| **20. MRI Metadata** | Patient ID, Modality, Volume Shape, Voxel Spacing |
| **21. Preprocessing Viewer** | Z-slider: Original vs Stage 2 Preprocessed |
| **22. Enhancement Viewer** | Z-slider: Stage 2 vs Stage 3 Enhanced |
| **23. Segmentation Viewer** | Enhanced volume + red contour outlines + clinical arrows |
| **24. Disease Detection** | Color-coded Normal/Pathological banner + diagnosis |
| **25. Measurements** | Volume, Max Area, Affected Tissue % |
| **26. Confidence Analysis** | Classification confidence + probability distribution |
| **27. Heatmap Viewer** | Grad-CAM attention map overlay (Gaussian-blurred + Jet colormap) |
| **28. Clinical Report** | Generate final PDF report |
| **29. Downloads** | Download clinical PDF report |

### Dashboard Components

| Component | File | Description |
|-----------|------|-------------|
| `render_metric_card()` | `metrics_cards.py` | HTML metric card with before/after + delta |
| `display_side_by_side()` | `image_comparison.py` | 3-panel: original + processed + difference heatmap |
| `display_stage_grid()` | `image_comparison.py` | N-column pipeline stage grid |
| `create_histogram_comparison()` | `plotly_charts.py` | Overlay voxel intensity histograms |
| `create_radar_chart()` | `plotly_charts.py` | 6-axis normalized radar (PSNR, SSIM, UQI, FSIM, Entropy, Contrast) |
| `create_metrics_boxplot()` | `plotly_charts.py` | Boxplot across Brain vs Spine |
| `create_scatter_plot()` | `plotly_charts.py` | Two-metric scatter with hover data |

---

## Spine Pipeline

**Entry Point:** `python spine_main.py`

### Train/Test Split

| Split | Normal | Pathological | Total |
|-------|--------|-------------|-------|
| Training | SP7, SP3, SP2, SP8, SP5 | SP17, SP19, SP15, SP18, SP12 | 10 |
| Test | SP6, SP9, SP4, SP1, SP10 | SP22, SP21, SP11, SP20, SP23 | 10 |
| **Total** | 10 | 10 | **20** |

- **Split Method:** Random shuffle with `seed=42`, first 5 per class → train, rest → test
- **Output:** `scripts/spine_dataset_split.csv`

### Spine-Specific Modalities

T1W (TSE), T2W (TSE), T2W (DRIVE HR), T1W (CLEAR), STIR, SPAIR, T1W_GADO (3D TRA FS GD), MobiView Survey

---

## PDF Report Generation

### Stage 2 Preprocessing PDF (`generate_stage2_pdf.py`)

A **15-section** PDF report documenting the preprocessing justification:

1. Cover Page
2. Table of Contents
3. Executive Summary
4. Pipeline Overview (8-step flowchart)
5. Brain MRI Preprocessing Justification
6. Spine MRI Preprocessing Justification
7. Resizing & Scaling Analysis
8. Denoising Filter Comparison (Gaussian, Median, Bilateral, NLM)
9. N4 Bias Field Correction
10. CLAHE Contrast Enhancement
11. Skull Stripping
12. Data Augmentation Examples
13. Annotation Visualization
14. 17 Quality Metrics (Brain vs Spine comparison)
15. Per-Modality Analysis + Curated Dataset Summary

### Hackathon Evaluation PDF (`generate_hackathon_pdf.py`)

An **11-section** PDF for hackathon judges:

1. Executive Summary
2. Training Brain Dataset Statistics
3. Training Spine Dataset Statistics
4. Test Brain Dataset Statistics
5. Test Spine Dataset Statistics
6. Image Property Definitions
7. Cross-Dataset Comparison
8. Per-Volume Tables (Brain + Spine)
9. Visualizations (heatmaps, boxplots, overview panels)
10. Training vs Test Analysis
11. Conclusion

---

## Installation & Setup

### Prerequisites

- Python 3.13+
- CUDA-capable GPU (recommended for Stage 3 training)

### Install Dependencies

```bash
pip install -r project/requirements.txt
```

### Key Dependencies

```bash
pip install nibabel>=4.0 SimpleITK>=2.2 opencv-python>=4.6 scikit-image>=0.19
pip install numpy>=1.21 scipy>=1.8 pandas>=1.4 matplotlib>=3.5
pip install plotly>=5.10 streamlit>=1.20 reportlab>=3.6
pip install torch>=2.0 torchvision>=0.15 pyyaml>=6.0 tqdm tabulate
```

---

## Usage

### Stage 1 — Dataset Analysis

```bash
# Brain MRI analysis
python project/main.py

# Spine MRI analysis
python project/spine_main.py
```

### Stage 2 — Preprocessing

```bash
# Run the 8-step preprocessing pipeline
python scripts/stage2_pipeline.py

# Generate Stage 2 justification PDF
python project/generate_stage2_pdf.py
```

### Stage 3 — AI Enhancement

```bash
# Full pipeline: train → evaluate → visualize → report
python project/run_stage3.py
```

### Stage 4 — Segmentation & Reports

```bash
# Via dashboard upload station or:
python scripts/stage4_inference.py
```

### Dashboard

```bash
# Launch Streamlit dashboard (29 pages)
python project/run_dashboard.py
```

### Hackathon PDF Report

```bash
# Generate comprehensive evaluation PDF
python project/generate_hackathon_pdf.py
```

---

## Quality Metrics Reference

### Full-Reference Metrics (require ground truth)

| Metric | Range | Best | Measures |
|--------|-------|------|----------|
| PSNR | [0, ∞) dB | ↑ | Signal fidelity |
| SSIM | [0, 1] | ↑ | Structural similarity |
| MSE | [0, ∞) | ↓ | Pixel error |
| RMSE | [0, ∞) | ↓ | Root pixel error |
| UQI | [0, 1] | ↑ | Universal quality |
| FSIM | [0, 1] | ↑ | Feature similarity |
| LPIPS | [0, ∞) | ↓ | Perceptual distance |

### No-Reference/Blind Metrics

| Metric | Range | Best | Measures |
|--------|-------|------|----------|
| GMSD | [0, ∞) | ↓ | Gradient consistency |
| VIF | [0, ∞) | ↑ | Visual information |
| BRISQUE | [0, 100] | ↓ | Natural scene statistics |
| NIQE | [0, ∞) | ↓ | Natural image quality |
| PIQE | [0, 100] | ↓ | Perceptual quality |

### Statistical Metrics

| Metric | Measures |
|--------|----------|
| Entropy | Information content (Shannon, 256-bin) |
| Contrast | RMS contrast of foreground |
| Sharpness | Laplacian variance |
| Edge Strength | Sobel gradient magnitude |
| Noise Level | Background MAD / 0.6745 |

---

## License

Academic project — MedhaDrishti National-Level AI Hackathon, Yugma TechFest 2.0.
