# STAGE 1: SPINE MRI DATASET EXPLORATION, ANALYSIS AND PREPARATION REPORT
## MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)

**Challenge:** AI for Medical Image Enhancement and Segmentation  
**Dataset:** Spine MRI Training Dataset (`training_data_spine/`)  
**Task:** Binary Classification — Normal vs Pathological Spine MRI  
**Modalities:** T1W, T2W, STIR, Gadolinium-Enhanced, MobiView Survey  

---

## 1. Introduction
Spinal MRI is the gold-standard non-invasive diagnostic modality for evaluating
vertebral pathologies, disc herniations, spinal cord compression, tumors,
inflammatory conditions, and degenerative changes. Multi-sequence spine MRI
acquisition provides critical soft-tissue contrast for identifying normal
anatomical structures (vertebral bodies, intervertebral discs, spinal cord,
nerve roots) and pathological findings (disc protrusions, cord signal changes,
enhancing lesions, edema).

This report presents **Stage 1 (Dataset Exploration, Analysis, and Preparation)**
for the Spine MRI classification pipeline. Stage 1 focuses on comprehensive
dataset discovery, NIfTI voxel property assessment, multi-sequence intensity
metrics, spatial resolution analysis, image quality quantification, quality
control audits, Normal vs Pathological comparison, and dataset summaries.

---

## 2. Dataset Description
The analysis utilizes a custom Spine MRI dataset stored inside
`training_data_spine/`, consisting of Normal and Pathological patient folders
extracted from the complete Spine DATASETS collection.

- **Total Patients Analyzed:** 10 Patients
  - Normal Patients: 5
  - Pathological Patients: 5
- **Total NIfTI MRI Volumes:** 186 Volumes
- **File Format:** NIfTI-1 compressed (`.nii.gz`)
- **Modality Sequences per Patient:**
  1. **T1-Weighted (T1W / eT1W):** T1-weighted spin-echo sequences with
     optional CLEAR homogeneity correction; anatomical detail imaging.
  2. **T2-Weighted (T2W / eT2W):** T2-weighted spin-echo sequences including
     DRIVE high-resolution variants; fluid-sensitive imaging.
  3. **STIR (Short Tau Inversion Recovery):** Fat-suppressed T2-sensitive
     sequence for edema and inflammation detection.
  4. **Gadolinium-Enhanced T1W (T1W_GADO):** Post-contrast T1 sequences for
     active lesion and tumor enhancement detection (Pathological cases).
  5. **Survey / MobiView:** Quick localizer scans for anatomical orientation.
  6. **SPAIR / Special:** Spectral Adiabatic Inversion Recovery sequences
     (variant fat suppression technique).
- **Classification Task:** Binary — Normal (healthy control) vs
  Pathological (spinal pathology confirmed)

---

## 3. Folder Structure
```
training_data_spine/
├── Normal Spine MRI Datasets/
│   ├── SP2/   (16 files)
│   ├── SP3/   (34 files)
│   ├── SP5/   (5 files)
│   ├── SP7/   (19 files)
│   └── SP8/   (16 files)
│
├── Pathological Spine MRI Datasets/
│   ├── SP12/  (34 files)
│   ├── SP15/  (11 files)
│   ├── SP17/  (17 files)
│   ├── SP18/  (21 files)
│   └── SP19/  (13 files)
│
├── analysis/          (Statistics CSVs)
├── figures/           (Publication-quality PNGs)
├── reports/           (Stage1_Spine_Report.md / .pdf)
├── scripts/           (Pipeline Python modules)
├── notebooks/         (Interactive notebook)
└── spine_main.py      (Pipeline entry point)
```

---

## 4. Dataset Statistics
The automated pipeline evaluated the entire dataset. Key global statistics
are summarized below:

| Metric                              | Value                                                  |
|:------------------------------------|:-------------------------------------------------------|
| Total Patients                      | 10                                                     |
| Total Normal Patients               | 5                                                      |
| Total Pathological Patients         | 5                                                      |
| Total MRI Volumes                   | 186                                                    |
| Number of T1W Scans                 | 76                                                     |
| Number of T1W GADO Scans            | 6                                                      |
| Number of T2W Scans                 | 88                                                     |
| Number of STIR Scans                | 13                                                     |
| Number of Survey/Locator Scans      | 2                                                      |
| Other/Special Sequences             | 1                                                      |
| Average File Size (MB)              | 1.26                                                   |
| Largest File Name                   | S82170_MobiView_eT1W_TSE_20260221100326_405.nii.gz     |
| Largest File Size (MB)              | 7.47                                                   |
| Smallest File Name                  | S82030_eT1W_TSE_CLEAR_20260305081552_702_i00001.nii.gz |
| Smallest File Size (MB)             | 0.06                                                   |
| Total Dataset Size (MB)             | 233.98                                                 |
| Total Dataset Size (GB)             | 0.228                                                  |
| Average SNR (Excl. Survey)          | 59127946.5653                                          |
| Average RMS Contrast (Excl. Survey) | 0.7044                                                 |
| Average Entropy (Excl. Survey)      | 6.8306                                                 |

![Dataset Completeness Status](../figures/quality_analysis/quality_checks_summary.png)

---

## 5. MRI Modalities & Clinical Significance

| Modality | Clinical Purpose & Significance |
| :--- | :--- |
| **T1-Weighted (T1W)** | Delineates vertebral anatomy, bone marrow composition, disc height, and
spinal cord gray-white matter differentiation. Dark CSF, bright fat. |
| **T2-Weighted (T2W)** | Highly sensitive to fluid and water content. Brightly visualizes CSF,
disc hydration, edema, and cord signal changes. DRIVE variants provide
enhanced high-resolution imaging. |
| **STIR** | Fat-suppressed sequence sensitive to edema and inflammation. Critical
for detecting bone marrow edema, facet joint inflammation, and soft-tissue
pathology. |
| **T1W GADO** | Post-gadolinium contrast T1 sequence for detecting blood-spinal cord
barrier disruption, tumor enhancement, and active inflammatory lesions. |
| **Survey/MobiView** | Quick localizer scans providing anatomical overview for planning detailed
sequence acquisitions. |

![Modality Comparison — Normal](../figures/modality_comparison/normal_modality_comparison.png)
![Modality Comparison — Pathological](../figures/modality_comparison/pathological_modality_comparison.png)

---

## 6. Image Dimension Analysis
Spatial geometry was verified across all 186 NIfTI files:

- Volume dimensions vary across patients and modalities due to heterogeneous
  scan protocols (different field-of-view, slice thickness, and matrix sizes).
- Multiple unique dimension variants detected, reflecting real-world clinical
  acquisition variability.
- The dataset includes both high-resolution 3D volumes and rapid 2D scout scans.

![Dimensions Distribution](../figures/resolution_analysis/image_dimensions_distribution.png)

---

## 7. Voxel Analysis
Voxel grid dimensions vary by acquisition protocol:

- Voxel spacing is heterogeneous across patients and sequences, reflecting
  different clinical scanner protocols (1.5T vs 3T acquisition parameters).
- Typical voxel resolutions range from sub-millimeter to several millimeters
  depending on the scan type (survey vs detailed diagnostic sequences).

![Voxel Spacing Distribution](../figures/resolution_analysis/voxel_spacing_distribution.png)

---

## 8. Intensity Analysis
Intensity distributions vary significantly across sequences due to different
pulse sequence dynamics:

| Modality   |   Volume_Count |   Patients_With_Modality |   Average_Intensity |   Median_Intensity |   Min_Intensity |   Max_Intensity |   Average_Contrast |   Average_Entropy |   Average_Sharpness |   Average_Noise |   Average_SNR |   Average_Edge_Strength | Typical_Dimensions   | Typical_Voxel_Spacing    |   Average_File_Size_MB |
|:-----------|---------------:|-------------------------:|--------------------:|-------------------:|----------------:|----------------:|-------------------:|------------------:|--------------------:|----------------:|--------------:|------------------------:|:---------------------|:-------------------------|-----------------------:|
| SPAIR      |              1 |                        1 |              343.72 |             264.18 |               0 |         2588.94 |             0.7162 |            5.8904 |            0.001693 |          0      |   3.43724e+07 |                0.155537 | 432 x 432 x 17       | 0.694 x 0.694 x 5.0 mm   |                   2.99 |
| STIR       |             13 |                        7 |              445.45 |             362.2  |               0 |         3572.87 |             0.7153 |            6.1456 |            0.002787 |          0      |   4.45447e+07 |                0.117713 | 432 x 432 x 15       | 0.865 x 0.865 x 4.4 mm   |                   2.87 |
| Survey     |              2 |                        2 |              520.26 |             298.14 |               0 |         2771.27 |             0.7917 |            4.9676 |            0.002164 |          0      |   5.20265e+07 |                0.090684 | 528 x 1017 x 5       | 0.758 x 0.758 x 8.997 mm |                   2.09 |
| T1W        |             76 |                       10 |              709.55 |             617.62 |               0 |         4559.28 |             0.6286 |            6.9616 |            0.005476 |         12.2166 |   5.79925e+07 |                0.207618 | 224 x 224 x 3        | 0.714 x 0.714 x 4.4 mm   |                   0.99 |
| T1W_GADO   |              6 |                        1 |              372.77 |             339.42 |               0 |         7171.26 |             0.6954 |            5.7188 |            0.004155 |          0      |   3.72765e+07 |                0.140263 | 224 x 224 x 20       | 0.893 x 0.893 x 4.4 mm   |                   1.02 |
| T2W        |             88 |                       10 |              640.34 |             549.87 |               0 |         4252.8  |             0.7687 |            6.9053 |            0.008701 |          0      |   6.40341e+07 |                0.224974 | 288 x 288 x 5        | 0.556 x 0.556 x 4.4 mm   |                   1.23 |

![Intensity Histograms](../figures/histograms/intensity_histograms.png)
![Intensity Distribution Overlay](../figures/histograms/intensity_distribution_overlay.png)
![Intensity by Class](../figures/histograms/intensity_by_class.png)

---

## 9. Contrast Analysis
Root-Mean-Square (RMS contrast measures tissue signal variability:

- T1W sequences provide moderate contrast between vertebral bone marrow,
  disc material, and CSF.
- T2W sequences provide high contrast between CSF (bright) and disc/cord.
- STIR sequences provide targeted fat-suppressed contrast for edema detection.

![Contrast Boxplot](../figures/boxplots/contrast_boxplot.png)

---

## 10. Entropy Analysis
Shannon Information Entropy quantifies information content and textural
complexity across modality volumes:

- Higher entropy values indicate greater textural diversity and potential
  pathological heterogeneity.
- Pathological patients may exhibit elevated entropy in affected regions.

![Entropy Boxplot](../figures/boxplots/entropy_boxplot.png)

---

## 11. Noise Analysis
Background noise standard deviation and MAD signal quality estimates:

![Noise Boxplot](../figures/boxplots/noise_boxplot.png)
![SNR Distribution](../figures/quality_analysis/snr_distribution.png)

---

## 12. Sharpness Analysis
3D Laplacian spatial variance measures high-frequency detail and edge
crispness across spine volumes:

![Sharpness Boxplot](../figures/boxplots/sharpness_boxplot.png)

---

## 13. Normal vs Pathological Comparison
A critical aspect of this dataset is the binary classification between
Normal and Pathological spine MRI scans. Key observations:

| Class        |   Patient_Count |   Volume_Count |   Mean_Intensity_Avg |   Contrast_Avg |   Entropy_Avg |   Sharpness_Avg |     SNR_Avg |   Noise_Avg |   Edge_Strength_Avg |   File_Size_MB_Avg |   Dimensions_Variants |
|:-------------|----------------:|---------------:|---------------------:|---------------:|--------------:|----------------:|------------:|------------:|--------------------:|-------------------:|----------------------:|
| Normal       |               5 |             89 |               621.27 |         0.7397 |        6.9191 |        0.006916 | 6.11232e+07 |      0.7902 |            0.208274 |               1.12 |                    30 |
| Pathological |               5 |             95 |               666.88 |         0.6713 |        6.7478 |        0.006624 | 5.72587e+07 |      9.033  |            0.205975 |               1.37 |                    38 |

![Normal vs Pathological Boxplots](../figures/class_comparison/normal_vs_pathological_boxplots.png)
![Class Bar Comparison](../figures/class_comparison/class_bar_comparison.png)

---

## 14. Dataset Challenges
1. **Heterogeneous Acquisitions:** Variable scanner protocols, matrix sizes,
   and voxel resolutions across patients require robust preprocessing
   (resampling, normalization) before deep learning modeling.
2. **Multi-Protocol Per Patient:** Each patient contains multiple sequences
   (T1W, T2W, STIR, etc.) requiring careful sequence-level data loading
   strategies.
3. **Limited Patient Count:** With 10 patients (5 Normal,
   5 Pathological), data augmentation and transfer learning strategies
   are essential.
4. **Class-Specific Sequences:** Gadolinium contrast (GADO) sequences are
   only present in pathological cases, potentially introducing confounding
   features.

---

## 15. Important Observations
1. **All 10 patients** successfully processed with zero corrupted NIfTI files.
2. **Core modalities (T1W + T2W)** present in all patients, confirming
   acquisition completeness for basic diagnostic capability.
3. **Variable file counts per patient** (5 to 34 files) reflect differences
   in clinical acquisition protocols.
4. **Heterogeneous dimensions and spacings** necessitate standardization
   during preprocessing.
5. **Patient SP17** uniquely contains gadolinium contrast-enhanced sequences,
   confirming its pathological classification with active contrast enhancement.

---

## 16. Conclusion
The Spine MRI training dataset (`training_data_spine/`) comprises
**10 patients (5 Normal, 5 Pathological)**
with a total of **186 NIfTI volumes** spanning T1W, T2W, STIR,
Gadolinium-enhanced, and Survey sequences. The dataset exhibits heterogeneous
but clinically representative acquisition parameters.

Stage 1 analysis confirms the dataset is fully validated, cataloged, and ready
for Stage 2 preprocessing (intensity normalization, spatial resampling, data
augmentation) and Stage 3 deep learning model development for binary spine
pathology classification.

**Total Dataset Size:** 0.228 GB  
**Average SNR (Excl. Survey):** 59127946.5653  
**Average RMS Contrast:** 0.7044  
**Average Entropy:** 6.8306 bits  

---
*Report automatically generated by Stage 1 Spine Pipeline for MedhaDrishti AI Hackathon.*
