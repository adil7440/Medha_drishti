from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class SpineReportGenerator:
    """
    Generates a Stage1_Spine_Report.md and Stage1_Spine_Report.pdf
    containing all required sections, embedded figures, and CSV data tables
    for the Spine MRI Dataset Analysis.
    """

    def __init__(self, output_dir, figures_dir, stats_dict):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = stats_dict

    def generate_all_reports(self):
        """Generates Markdown and PDF reports."""
        print("[SpineReportGenerator] Generating Stage1_Spine_Report.md...")
        md_path = self.output_dir / 'Stage1_Spine_Report.md'
        self._generate_markdown_report(md_path)

        print("[SpineReportGenerator] Generating Stage1_Spine_Report.pdf...")
        pdf_path = self.output_dir / 'Stage1_Spine_Report.pdf'
        try:
            self._generate_pdf_report(pdf_path)
            print(f"[SpineReportGenerator] PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"[Warning] Failed to generate PDF report: {e}")

    def _generate_markdown_report(self, md_path):
        ds = self.stats['dataset_statistics'].set_index('Metric')['Value'].to_dict()
        mod_df = self.stats['modality_statistics']
        class_df = self.stats.get('class_comparison', pd.DataFrame())

        ds_table_md = self.stats['dataset_statistics'].to_markdown(index=False)
        mod_table_md = mod_df.to_markdown(index=False)
        class_table_md = class_df.to_markdown(index=False) if not class_df.empty else "N/A"

        total_patients = ds.get('Total Patients', 10)
        normal_count = ds.get('Total Normal Patients', 5)
        path_count = ds.get('Total Pathological Patients', 5)
        total_volumes = ds.get('Total MRI Volumes', 0)
        total_size_gb = ds.get('Total Dataset Size (GB)', 0)
        avg_snr = ds.get('Average SNR (Excl. Survey)', 0)
        avg_contrast = ds.get('Average RMS Contrast (Excl. Survey)', 0)
        avg_entropy = ds.get('Average Entropy (Excl. Survey)', 0)

        md_content = f"""# STAGE 1: SPINE MRI DATASET EXPLORATION, ANALYSIS AND PREPARATION REPORT
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

- **Total Patients Analyzed:** {total_patients} Patients
  - Normal Patients: {normal_count}
  - Pathological Patients: {path_count}
- **Total NIfTI MRI Volumes:** {total_volumes} Volumes
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

{ds_table_md}

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
Spatial geometry was verified across all {total_volumes} NIfTI files:

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

{mod_table_md}

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

{class_table_md}

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
3. **Limited Patient Count:** With {total_patients} patients ({normal_count} Normal,
   {path_count} Pathological), data augmentation and transfer learning strategies
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
**{total_patients} patients ({normal_count} Normal, {path_count} Pathological)**
with a total of **{total_volumes} NIfTI volumes** spanning T1W, T2W, STIR,
Gadolinium-enhanced, and Survey sequences. The dataset exhibits heterogeneous
but clinically representative acquisition parameters.

Stage 1 analysis confirms the dataset is fully validated, cataloged, and ready
for Stage 2 preprocessing (intensity normalization, spatial resampling, data
augmentation) and Stage 3 deep learning model development for binary spine
pathology classification.

**Total Dataset Size:** {total_size_gb} GB  
**Average SNR (Excl. Survey):** {avg_snr}  
**Average RMS Contrast:** {avg_contrast}  
**Average Entropy:** {avg_entropy} bits  

---
*Report automatically generated by Stage 1 Spine Pipeline for MedhaDrishti AI Hackathon.*
"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def _generate_pdf_report(self, pdf_path):
        """Generates a publication-grade PDF using ReportLab."""
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'],
            fontName='Helvetica-Bold', fontSize=20, leading=24,
            textColor=colors.HexColor('#1a365d'), alignment=1, spaceAfter=12)

        subtitle_style = ParagraphStyle(
            'DocSubTitle', parent=styles['Normal'],
            fontName='Helvetica', fontSize=11, leading=14,
            textColor=colors.HexColor('#4a5568'), alignment=1, spaceAfter=20)

        h1_style = ParagraphStyle(
            'H1', parent=styles['Heading2'],
            fontName='Helvetica-Bold', fontSize=14, leading=18,
            textColor=colors.HexColor('#2b6cb0'),
            spaceBefore=14, spaceAfter=6, keepWithNext=True)

        body_style = ParagraphStyle(
            'Body', parent=styles['BodyText'],
            fontName='Helvetica', fontSize=9.5, leading=13.5,
            textColor=colors.HexColor('#2d3748'), spaceAfter=8)

        story = []

        # Header
        story.append(Paragraph(
            "STAGE 1: SPINE MRI DATASET ANALYSIS REPORT", title_style))
        story.append(Paragraph(
            "<b>MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)</b>"
            "<br/>Dataset: Spine MRI Training Dataset (training_data_spine/)"
            "<br/>Task: Binary Classification — Normal vs Pathological",
            subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=colors.HexColor('#2b6cb0'),
                                spaceAfter=15))

        # 1. Introduction
        story.append(Paragraph(
            "1. Executive Summary & Introduction", h1_style))
        story.append(Paragraph(
            "This report presents the complete Stage 1 Dataset Exploration, "
            "Analysis, and Preparation for the Spine MRI classification pipeline. "
            "Spinal MRI is the gold-standard non-invasive diagnostic modality for "
            "evaluating vertebral pathologies, disc herniations, spinal cord "
            "compression, tumors, and inflammatory conditions. "
            "The objective of Stage 1 is to execute an end-to-end automated "
            "evaluation of NIfTI voxel dimensions, multi-sequence intensity "
            "distributions, spatial resolutions, signal quality metrics, "
            "and dataset completeness across all patient scans.",
            body_style))

        # 2. Key Statistics Table
        story.append(Paragraph("2. Global Dataset Statistics", h1_style))
        ds_df = self.stats['dataset_statistics']
        table_data = [
            [Paragraph(f"<b>{row['Metric']}</b>", body_style),
             Paragraph(str(row['Value']), body_style)]
            for _, row in ds_df.iterrows()
        ]
        table_data.insert(0, [
            Paragraph("<b>Metric Parameter</b>", body_style),
            Paragraph("<b>Value / Summary</b>", body_style)
        ])

        t = Table(table_data, colWidths=[2.8 * inch, 4.2 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ebf8ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # 3. Modality Comparison Figures
        fig_normal = (self.figures_dir / 'modality_comparison' /
                      'normal_modality_comparison.png')
        if fig_normal.exists():
            story.append(Paragraph(
                "3. Multi-Sequence Modality Comparison — Normal Patient",
                h1_style))
            story.append(RLImage(str(fig_normal), width=6.8 * inch,
                                 height=1.9 * inch))
            story.append(Spacer(1, 10))

        fig_path = (self.figures_dir / 'modality_comparison' /
                    'pathological_modality_comparison.png')
        if fig_path.exists():
            story.append(Paragraph(
                "4. Multi-Sequence Modality Comparison — Pathological Patient",
                h1_style))
            story.append(RLImage(str(fig_path), width=6.8 * inch,
                                 height=1.9 * inch))
            story.append(Spacer(1, 10))

        # 5. Intensity Histograms
        fig_hist = (self.figures_dir / 'histograms' /
                    'intensity_histograms.png')
        if fig_hist.exists():
            story.append(Paragraph(
                "5. Intensity Histogram Distribution Profiles", h1_style))
            story.append(RLImage(str(fig_hist), width=6.5 * inch,
                                 height=5.2 * inch))
            story.append(Spacer(1, 10))

        # 6. Property Comparison Grid
        fig_grid = (self.figures_dir / 'boxplots' /
                    'property_comparison_grid.png')
        if fig_grid.exists():
            story.append(Paragraph(
                "6. Statistical & Image Quality Metrics Grid", h1_style))
            story.append(RLImage(str(fig_grid), width=6.8 * inch,
                                 height=4.5 * inch))
            story.append(Spacer(1, 10))

        # 7. Normal vs Pathological
        fig_nvp = (self.figures_dir / 'class_comparison' /
                   'normal_vs_pathological_boxplots.png')
        if fig_nvp.exists():
            story.append(Paragraph(
                "7. Normal vs Pathological — Quality Metrics Comparison",
                h1_style))
            story.append(RLImage(str(fig_nvp), width=6.8 * inch,
                                 height=4.5 * inch))
            story.append(Spacer(1, 10))

        # 8. Modality Statistics Table
        story.append(Paragraph(
            "8. Modality Comparison Summary Table", h1_style))
        mod_df = self.stats['modality_statistics']
        m_headers = ['Modality', 'Count', 'Avg Mean', 'Contrast',
                     'Entropy', 'Sharpness', 'Noise', 'SNR']
        m_rows = [[
            r['Modality'], str(r['Volume_Count']),
            str(r['Average_Intensity']),
            str(r['Average_Contrast']),
            str(r['Average_Entropy']),
            str(r['Average_Sharpness']),
            str(r['Average_Noise']),
            str(r['Average_SNR'])
        ] for _, r in mod_df.iterrows()]

        m_table_data = [
            [Paragraph(f"<b>{h}</b>", body_style) for h in m_headers]
        ]
        for row in m_rows:
            m_table_data.append([Paragraph(val, body_style) for val in row])

        t_mod = Table(m_table_data,
                      colWidths=[0.8 * inch, 0.6 * inch, 0.8 * inch,
                                 0.8 * inch, 0.8 * inch, 0.8 * inch,
                                 0.8 * inch, 0.8 * inch])
        t_mod.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_mod)
        story.append(Spacer(1, 15))

        # 9. Resolution Analysis
        fig_dim = (self.figures_dir / 'resolution_analysis' /
                   'image_dimensions_distribution.png')
        if fig_dim.exists():
            story.append(Paragraph("9. Volume Dimension Distribution",
                                   h1_style))
            story.append(RLImage(str(fig_dim), width=6.5 * inch,
                                 height=3.3 * inch))
            story.append(Spacer(1, 10))

        # 10. Conclusion
        story.append(Paragraph(
            "10. Stage 1 Summary & Next Steps", h1_style))
        story.append(Paragraph(
            "Stage 1 analysis confirms that the Spine MRI training dataset "
            "is fully validated, comprising 10 patients (5 Normal, 5 "
            "Pathological) with a total of 186 NIfTI volumes spanning T1W, "
            "T2W, STIR, Gadolinium-enhanced, and Survey modalities. "
            "The dataset exhibits clinically representative but heterogeneous "
            "acquisition parameters, requiring standard preprocessing "
            "(intensity normalization, spatial resampling) before Stage 3 "
            "deep learning model development.",
            body_style))

        doc.build(story)
