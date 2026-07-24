import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / 'scripts'
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.spine_dataset_loader import SpineDatasetLoader
from scripts.spine_image_properties import SpineImagePropertyExtractor
from scripts.spine_statistics import SpineDatasetStatisticsCalculator
from scripts.spine_visualization import SpineDatasetVisualizer
from scripts.spine_report_generator import SpineReportGenerator


def process_single_volume_task(task_args):
    """Worker function for parallel NIfTI volume property extraction."""
    patient_id, file_path = task_args
    try:
        return SpineImagePropertyExtractor.extract_properties(patient_id, file_path)
    except Exception as e:
        print(f"[Error] Failed to process {patient_id} {Path(file_path).name}: {e}")
        return None


def main():
    print("=" * 80)
    print(" STAGE 1: SPINE MRI DATASET EXPLORATION, ANALYSIS & PREPARATION PIPELINE")
    print(" MedhaDrishti National-Level AI Hackathon (Yugma TechFest 2.0)")
    print("=" * 80)

    start_time = time.time()

    # 1. Directory Setup
    dataset_dir = PROJECT_DIR.parent / 'training_data_spine'
    if not dataset_dir.exists():
        dataset_dir = PROJECT_DIR / 'training_data_spine'

    analysis_dir = PROJECT_DIR / 'analysis' / 'spine'
    figures_dir = PROJECT_DIR / 'figures' / 'spine'
    reports_dir = PROJECT_DIR / 'reports' / 'spine'
    notebooks_dir = PROJECT_DIR / 'notebooks'

    for d in [analysis_dir, figures_dir, reports_dir, notebooks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1/7] Discovering spine dataset in: {dataset_dir}")
    loader = SpineDatasetLoader(dataset_dir)
    patient_df = loader.scan_dataset()
    print(f" -> Discovered {len(patient_df)} patient folders.")
    print(f"    Normal: {len(patient_df[patient_df['Class'] == 'Normal'])}")
    print(f"    Pathological: {len(patient_df[patient_df['Class'] == 'Pathological'])}")

    if patient_df.empty:
        print("[Error] No patient folders found in dataset directory!")
        return

    # Save patient inventory
    patient_df.to_csv(analysis_dir / 'spine_patient_inventory.csv', index=False)

    # 2. Prepare parallel tasks for all volumes
    tasks = []
    for _, row in patient_df.iterrows():
        pid = row['Patient_ID']
        p_dir = Path(row['Patient_Dir'])
        nii_files = list(p_dir.glob("*.nii.gz")) + list(p_dir.glob("*.nii"))
        for f in nii_files:
            tasks.append((pid, str(f)))

    print(f"\n[Step 2/7] Extracting image properties for {len(tasks)} MRI NIfTI volumes...")
    properties_records = []

    num_workers = max(1, os.cpu_count() - 1)
    print(f" -> Utilizing {num_workers} parallel CPU workers.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_volume_task, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="Analyzing Spine Volumes"):
            res = f.result()
            if res is not None:
                properties_records.append(res)

    properties_df = pd.DataFrame(properties_records)
    print(f" -> Extraction complete. Processed {len(properties_df)} volumes.")

    # 3. Calculate Statistics & Quality Checks
    print(f"\n[Step 3/7] Generating statistics CSVs and performing quality checks...")
    calculator = SpineDatasetStatisticsCalculator(
        patient_df, properties_df, analysis_dir)
    stats_dict = calculator.generate_all_statistics()

    qc_results = calculator.run_quality_checks()
    print("\n--- SPINE DATASET QUALITY CHECKS SUMMARY ---")
    for k, v in qc_results['summary'].items():
        print(f"  [OK] {k}: {v}")
    if qc_results['warnings']:
        print("  [WARNINGS]:")
        for w in qc_results['warnings']:
            print(f"    ! {w}")
    else:
        print("  [OK] ZERO DATA QUALITY WARNINGS DETECTED!")
    print("---------------------------------------------")

    # 4. Generate Visualizations
    print(f"\n[Step 4/7] Generating publication-quality figures...")
    visualizer = SpineDatasetVisualizer(properties_df, patient_df, figures_dir)
    visualizer.generate_all_figures()

    # 5. Generate Reports
    print(f"\n[Step 5/7] Generating Stage 1 spine reports (Markdown & PDF)...")
    report_gen = SpineReportGenerator(reports_dir, figures_dir, stats_dict)
    report_gen.generate_all_reports()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" STAGE 1 SPINE ANALYSIS PIPELINE COMPLETED IN {elapsed:.2f} SECONDS!")
    print(" Output Deliverables Generated:")
    print(f"  |-- Analysis CSVs:    {analysis_dir}")
    print(f"  |-- Figures (PNG):    {figures_dir}")
    print(f"  |-- Report (MD):      {reports_dir / 'Stage1_Spine_Report.md'}")
    print(f"  |-- Report (PDF):     {reports_dir / 'Stage1_Spine_Report.pdf'}")
    print(f"  '-- Patient Inventory:{analysis_dir / 'spine_patient_inventory.csv'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
