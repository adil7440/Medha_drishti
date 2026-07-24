import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.train_stage3 import run_stage3_training
from scripts.evaluate_stage3 import run_stage3_evaluation
from scripts.stage3_report_generator import Stage3ReportGenerator


def main():
    print("=" * 80)
    print(" STAGE 3: AI-BASED MRI ENHANCEMENT PIPELINE")
    print(" MedhaDrishti National-Level AI Hackathon")
    print("=" * 80)

    start_time = time.time()

    preprocessed_dir = PROJECT_DIR / "stage2" / "preprocessed"
    stage3_dir = PROJECT_DIR / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train DnCNN & SwinIR Small (5 Epochs for fast execution)
    print("\n[Step 1/3] Training DnCNN and SwinIR Small models on GPU/CPU...")
    run_stage3_training(str(preprocessed_dir), max_epochs=5)

    # 2. Benchmark & Leaderboard Ranking
    print("\n[Step 2/3] Evaluating models and generating Leaderboard...")
    leaderboard_df = run_stage3_evaluation(str(preprocessed_dir))

    # 3. Report Generation
    print("\n[Step 3/3] Exporting Stage 3 Technical Report...")
    csv_p = stage3_dir / "metrics" / "stage3_model_comparison.csv"
    report_p = stage3_dir / "reports" / "stage3_preprocessing_report.md"
    Stage3ReportGenerator.generate_report(str(csv_p), str(report_p))

    print(f"\n[Success] Stage 3 Pipeline Complete in {round(time.time() - start_time, 2)}s!")


if __name__ == "__main__":
    main()
