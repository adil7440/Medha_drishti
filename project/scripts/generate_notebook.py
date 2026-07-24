import nbformat as nbf
from pathlib import Path

def create_stage1_notebook(notebook_path):
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# STAGE 1: DATASET EXPLORATION, ANALYSIS AND PREPARATION
## MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)
**Topic:** AI for Medical Image Enhancement and Segmentation  
**Dataset:** BraTS 2020 Brain MRI Training Dataset (`training_data_brain/`)

---
### Notebook Purpose
This interactive notebook executes the Stage 1 dataset discovery, physical voxel property assessment, multi-sequence intensity metrics, quality verification, and visualization pipeline.
"""),
        nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.append(str(Path('../scripts').resolve()))

from dataset_loader import BrainDatasetLoader
from image_properties import ImagePropertyExtractor
from statistics import DatasetStatisticsCalculator
from visualization import DatasetVisualizer
from report_generator import ReportGenerator

print("Pipeline modules imported successfully!")
"""),
        nbf.v4.new_markdown_cell("""## Step 1: Discover & Scan Dataset
Automatically discover patient folders, modalities (T1, T1CE, T2, FLAIR, SEG), and check completeness status.
"""),
        nbf.v4.new_code_cell("""base_dir = Path('../../training_data_brain').resolve()
if not base_dir.exists():
    base_dir = Path('../training_data_brain').resolve()

loader = BrainDatasetLoader(base_dir)
patient_df = loader.scan_dataset()

print(f"Total Patients Discovered: {len(patient_df)}")
display(patient_df.head(10))
"""),
        nbf.v4.new_markdown_cell("""## Step 2 & 3 & 4: Image Property Extraction
Extract spatial dimensions, voxel spacing, statistical intensity metrics (Mean, Median, Contrast, Shannon Entropy, Sharpness, Noise MAD, SNR) across all volumes.
"""),
        nbf.v4.new_code_cell("""# Display sample image properties if image_properties.csv exists
props_csv = Path('../analysis/image_properties.csv')
if props_csv.exists():
    props_df = pd.read_csv(props_csv)
    print(f"Total NIfTI Volumes Analyzed: {len(props_df)}")
    display(props_df[['Patient_ID', 'Modality', 'Width', 'Height', 'Depth', 'Spacing_X', 'Mean_Intensity', 'Contrast', 'Entropy', 'SNR']].head(10))
"""),
        nbf.v4.new_markdown_cell("""## Step 5: Modality Statistics & Comparisons
Compare average metrics across T1, T1CE, T2, FLAIR, and Seg masks.
"""),
        nbf.v4.new_code_cell("""mod_csv = Path('../analysis/modality_statistics.csv')
if mod_csv.exists():
    mod_df = pd.read_csv(mod_csv)
    display(mod_df)
"""),
        nbf.v4.new_markdown_cell("""## Step 6: Dataset Visualizations
Preview key generated figures.
"""),
        nbf.v4.new_code_cell("""from IPython.display import Image, display

fig_path = Path('../figures/modality_comparison/modality_4panel_comparison.png')
if fig_path.exists():
    display(Image(filename=str(fig_path)))
"""),
        nbf.v4.new_markdown_cell("""## Step 7: Automated Quality Checks
Verify dataset integrity, missing modalities, shape consistency, and voxel resolution uniformity.
"""),
        nbf.v4.new_code_cell("""if props_csv.exists():
    calc = DatasetStatisticsCalculator(patient_df, props_df, '../analysis')
    checks = calc.run_quality_checks()
    print("Quality Checks Summary:")
    for k, v in checks['summary'].items():
        print(f" - {k}: {v}")
    if checks['warnings']:
        print("\\nWarnings:")
        for w in checks['warnings']:
            print(f" ! {w}")
    else:
        print("\\nPASSED: Zero quality warnings detected!")
""")
    ]

    nb.cells = cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    out_p = Path('../notebooks/Stage1_Analysis.ipynb')
    out_p.parent.mkdir(parents=True, exist_ok=True)
    create_stage1_notebook(out_p)
    print(f"Created notebook: {out_p}")
