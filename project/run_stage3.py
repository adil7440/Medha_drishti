import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.train_stage3 import run_stage3_training, load_config
from scripts.evaluate_stage3 import run_stage3_evaluation
from scripts.stage3_report_generator import Stage3ReportGenerator
from scripts.stage3_visualization import generate_comparison_visualizations, generate_edge_comparison
from scripts.training_visualization import generate_all_training_charts, generate_model_comparison_charts


def main():
    print("=" * 80)
    print(" STAGE 3: AI-BASED MRI ENHANCEMENT PIPELINE")
    print(" MedhaDrishti National-Level AI Hackathon")
    print("=" * 80)

    start_time = time.time()

    # Load master configuration
    config = load_config()
    preprocessed_dir = str(PROJECT_DIR / "stage2" / "preprocessed")
    stage3_dir = PROJECT_DIR / config.get("output", {}).get("base_dir", "stage3")
    stage3_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = stage3_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models_enabled = [name for name, cfg in config.get("models", {}).items() if cfg.get("enabled", True)]
    print(f"\nModels to train: {', '.join(models_enabled)}")
    print(f"Max Epochs: {config['training']['max_epochs']}")
    print(f"Early Stopping Patience: {config['training']['early_stopping']['patience']}")
    print(f"Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")

    # Step 1: Train all enabled models
    print(f"\n{'=' * 80}")
    print(" STEP 1/5: Training All Models")
    print(f"{'=' * 80}")
    run_stage3_training(preprocessed_dir, config)

    # Step 2: Evaluate and rank models
    print(f"\n{'=' * 80}")
    print(" STEP 2/5: Evaluating & Ranking Models")
    print(f"{'=' * 80}")
    leaderboard_df = run_stage3_evaluation(preprocessed_dir, config)

    # Step 3: Generate training visualization charts
    print(f"\n{'=' * 80}")
    print(" STEP 3/5: Generating Training Charts")
    print(f"{'=' * 80}")
    generate_all_training_charts(stage3_dir)
    csv_path = metrics_dir / "stage3_model_comparison.csv"
    if csv_path.exists():
        generate_model_comparison_charts(str(csv_path), stage3_dir)

    # Step 4: Generate comparison visualizations
    print(f"\n{'=' * 80}")
    print(" STEP 4/5: Generating MRI Comparison Visualizations")
    print(f"{'=' * 80}")
    try:
        generate_comparison_visualizations(preprocessed_dir, config)
        generate_edge_comparison(preprocessed_dir, config)
    except Exception as e:
        print(f"[Warning] Visualization generation error: {e}")

    # Step 5: Generate final report
    print(f"\n{'=' * 80}")
    print(" STEP 5/5: Generating Final Report")
    print(f"{'=' * 80}")
    report_p = stage3_dir / "reports" / "stage3_enhancement_report.md"
    Stage3ReportGenerator.generate_report(str(csv_path), str(report_p))

    total_time = round(time.time() - start_time, 2)
    print(f"\n{'=' * 80}")
    print(f" STAGE 3 COMPLETE — Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f" Best Model: {leaderboard_df.iloc[0]['Model'] if leaderboard_df is not None else 'N/A'}")
    print(f" Output: {stage3_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
