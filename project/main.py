import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# Ensure project root and scripts directory are in Python path
PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / 'scripts'
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.dataset_loader import BrainDatasetLoader
from scripts.image_properties import ImagePropertyExtractor
from scripts.statistics import DatasetStatisticsCalculator
from scripts.visualization import DatasetVisualizer
from scripts.report_generator import ReportGenerator
from scripts.generate_notebook import create_stage1_notebook


def process_single_volume_task(task_args):
    """Worker function for parallel NIfTI volume property extraction."""
    patient_id, modality, file_path = task_args
    try:
        return ImagePropertyExtractor.extract_properties(patient_id, modality, file_path)
    except Exception as e:
        print(f"[Error] Failed to process {patient_id} {modality}: {e}")
        return None


def main():
    print("=" * 80)
    print(" STAGE 1: BRAIN MRI DATASET EXPLORATION, ANALYSIS & PREPARATION PIPELINE")
    print(" MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)")
    print("=" * 80)

    start_time = time.time()

    # 1. Directory Setup
    dataset_dir = PROJECT_DIR.parent / 'training_data_brain'
    if not dataset_dir.exists():
        dataset_dir = PROJECT_DIR / 'training_data_brain'

    analysis_dir = PROJECT_DIR / 'analysis'
    figures_dir = PROJECT_DIR / 'figures'
    reports_dir = PROJECT_DIR / 'reports'
    notebooks_dir = PROJECT_DIR / 'notebooks'

    for d in [analysis_dir, figures_dir, reports_dir, notebooks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1/7] Discovering dataset in: {dataset_dir}")
    loader = BrainDatasetLoader(dataset_dir)
    patient_df = loader.scan_dataset()
    print(f" -> Discovered {len(patient_df)} patient folders.")

    if patient_df.empty:
        print("[Error] No patient folders found in dataset directory!")
        return

    # Prepare parallel tasks for all volumes
    tasks = []
    for _, row in patient_df.iterrows():
        pid = row['Patient_ID']
        if row['T1_Path']:
            tasks.append((pid, 'T1', row['T1_Path']))
        if row['T1CE_Path']:
            tasks.append((pid, 'T1CE', row['T1CE_Path']))
        if row['T2_Path']:
            tasks.append((pid, 'T2', row['T2_Path']))
        if row['FLAIR_Path']:
            tasks.append((pid, 'FLAIR', row['FLAIR_Path']))
        if row['Seg_Path']:
            tasks.append((pid, 'SEG', row['Seg_Path']))

    print(f"\n[Step 2/7] Extracting image properties for {len(tasks)} MRI NIfTI volumes...")
    properties_records = []
    
    # Process in parallel using CPU cores
    num_workers = max(1, os.cpu_count() - 1)
    print(f" -> Utilizing {num_workers} parallel CPU workers.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_volume_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Analyzing Volumes"):
            res = f.result()
            if res is not None:
                properties_records.append(res)

    properties_df = pd.DataFrame(properties_records)
    print(f" -> Extraction complete. Successfully processed {len(properties_df)} volumes.")

    # 3. Calculate Statistics & Quality Checks
    print(f"\n[Step 3/7] Generating statistics CSVs and performing quality checks...")
    calculator = DatasetStatisticsCalculator(patient_df, properties_df, analysis_dir)
    stats_dict = calculator.generate_all_statistics()

    qc_results = calculator.run_quality_checks()
    print("\n--- AUTOMATED DATASET QUALITY CHECKS SUMMARY ---")
    for k, v in qc_results['summary'].items():
        print(f"  [OK] {k}: {v}")
    if qc_results['warnings']:
        print("  [WARNINGS]:")
        for w in qc_results['warnings']:
            print(f"    ! {w}")
    else:
        print("  [OK] ZERO DATA QUALITY WARNINGS DETECTED!")
    print("------------------------------------------------")

    # 4. Generate Visualizations
    print(f"\n[Step 4/7] Generating publication-quality figures...")
    visualizer = DatasetVisualizer(properties_df, patient_df, figures_dir)
    visualizer.generate_all_figures()

    # 5. Generate Markdown & PDF Reports
    print(f"\n[Step 5/7] Generating Stage 1 reports (Markdown & PDF)...")
    report_gen = ReportGenerator(reports_dir, figures_dir, stats_dict)
    report_gen.generate_all_reports()

    # 6. Generate Interactive Jupyter Notebook
    print(f"\n[Step 6/7] Generating Stage 1 Interactive Notebook...")
    notebook_path = notebooks_dir / 'Stage1_Analysis.ipynb'
    create_stage1_notebook(notebook_path)
    print(f" -> Notebook created at: {notebook_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" STAGE 1 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(" Output Deliverables Generated:")
    print(f"  ├── Analysis CSVs:   {analysis_dir}")
    print(f"  ├── Figures (PNG):   {figures_dir}")
    print(f"  ├── Report (MD):     {reports_dir / 'Stage1_Report.md'}")
    print(f"  ├── Report (PDF):    {reports_dir / 'Stage1_Report.pdf'}")
    print(f"  └── Notebook (IPYNB):{notebook_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
