import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# Add project root and scripts to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from scripts.mri_validator import MRIValidator
from scripts.resampler import MRIResampler
from scripts.intensity_normalizer import IntensityNormalizer
from scripts.noise_remover import NoiseRemover
from scripts.contrast_enhancer import ContrastEnhancer
from scripts.skull_stripper import SkullStripper
from scripts.data_augmentor import MRIAugmentor
from scripts.quality_evaluator import QualityEvaluator
from scripts.stage2_report_generator import Stage2ReportGenerator


def apply_n4_2d(slice_2d: np.ndarray) -> np.ndarray:
    """Fast 2D N4 Bias Field Correction on a slice."""
    s_min, s_max = np.min(slice_2d), np.max(slice_2d)
    if s_max == s_min:
        return slice_2d

    norm = ((slice_2d - s_min) / (s_max - s_min + 1e-8)).astype(np.float32)
    sitk_img = sitk.GetImageFromArray(norm)
    mask_img = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30, 20, 10])
    
    try:
        corrected_sitk = corrector.Execute(sitk_img, mask_img)
        corrected_arr = sitk.GetArrayFromImage(corrected_sitk)
        return corrected_arr * (s_max - s_min) + s_min
    except Exception:
        return slice_2d


def process_single_volume(args):
    """
    Fast worker function to process an individual NIfTI MRI volume through Stage 2 pipeline.
    """
    dataset_name, patient_id, modality, file_path, output_dirs = args
    stage2_preprocessed_dir = Path(output_dirs["preprocessed"])
    
    start_t = time.time()

    # 1. Read NIfTI & Validate
    val_res = MRIValidator.validate_file(file_path)
    if not val_res["is_valid"] or val_res["corrupted"]:
        print(f"[Warning] Validation failed for {patient_id} {modality}: {val_res['error_message']}")
        return None

    try:
        nimg = nib.load(file_path)
        data = nimg.get_fdata().astype(np.float32)

        if len(data.shape) > 3:
            data = data[:, :, :, 0]

        is_spine = (dataset_name.lower() == "spine")

        # Select central slice for quality metrics & visual intermediate tracking
        mid_z = data.shape[2] // 2
        orig_slice = data[:, :, mid_z].copy()

        # 2. Resampling (Voxel Normalization on slice level)
        norm_slice_stage = IntensityNormalizer.min_max(orig_slice, 0.0, 1.0)

        # 3. Intensity Normalization (Percentile Clipping + Z-score)
        clipped_slice = IntensityNormalizer.percentile_clipping(norm_slice_stage, p_min=0.5, p_max=99.5)
        norm_slice = IntensityNormalizer.min_max(clipped_slice, 0.0, 1.0)

        # 4. Noise Removal (Gaussian, Median, Bilateral, NLM)
        denoise_gauss = NoiseRemover.gaussian_denoise(norm_slice, sigma=1.0)
        denoise_median = NoiseRemover.median_denoise(norm_slice, size=3)
        denoise_bilat = NoiseRemover.bilateral_denoise_slice(norm_slice)
        denoise_nlm = NoiseRemover.nlm_denoise_slice(norm_slice)

        # 5. Fast N4 Bias Field Correction (SimpleITK)
        slice_n4 = apply_n4_2d(denoise_bilat)

        # 6. Contrast Enhancement (CLAHE)
        slice_clahe = ContrastEnhancer.clahe_slice(slice_n4, clip_limit=2.0)
        
        # 7. Skull Stripping (Performed for Brain, skipped for Spine)
        if not is_spine:
            brain_mask = SkullStripper.extract_brain_mask(slice_clahe[:, :, np.newaxis])[:, :, 0]
            slice_stripped = slice_clahe.copy()
            slice_stripped[~brain_mask] = 0
        else:
            slice_stripped = slice_clahe.copy()

        # 8. Data Augmentation
        aug_rot = MRIAugmentor.rotate(slice_stripped, angle_deg=10.0)
        aug_flip = MRIAugmentor.flip(slice_stripped, axis=1)
        aug_gamma = MRIAugmentor.gamma_correction(slice_stripped, gamma=1.2)
        aug_final = MRIAugmentor.add_gaussian_noise(aug_gamma, std=0.01)

        final_slice = slice_stripped.copy()

        # 9. Compute Quality Metrics (17 Metrics)
        metrics_dict = QualityEvaluator.evaluate_pair(orig_slice, final_slice)
        metrics_dict["Dataset"] = dataset_name
        metrics_dict["Patient_ID"] = patient_id
        metrics_dict["Modality"] = modality
        metrics_dict["Processing_Time_Sec"] = round(time.time() - start_t, 3)

        # Save preprocessed slice representation (.npz for fast dashboard loading)
        out_filename = f"{dataset_name}_{patient_id}_{modality}_preprocessed.npz"
        np.savez_compressed(
            stage2_preprocessed_dir / out_filename,
            orig_slice=orig_slice,
            stage_norm=norm_slice_stage,
            stage_denoise_gauss=denoise_gauss,
            stage_denoise_median=denoise_median,
            stage_denoise_bilat=denoise_bilat,
            stage_denoise_nlm=denoise_nlm,
            stage_n4=slice_n4,
            stage_clahe=slice_clahe,
            stage_final=final_slice,
            aug_rot=aug_rot,
            aug_flip=aug_flip,
            aug_gamma=aug_gamma,
            aug_final=aug_final,
            volume_shape=data.shape
        )

        return metrics_dict

    except Exception as e:
        print(f"[Error] Processing failed for {patient_id} {modality}: {e}")
        return None


def run_stage2_pipeline(max_brain_patients: int = 30):
    print("=" * 80)
    print(" STAGE 2: MRI DATASET PRE-PROCESSING & QUALITY EVALUATION PIPELINE")
    print(" MedhaDrishti National-Level AI Hackathon")
    print("=" * 80)

    start_time = time.time()

    # Output directory setup
    stage2_dir = PROJECT_DIR / "stage2"
    dirs = {
        "preprocessed": stage2_dir / "preprocessed",
        "enhanced": stage2_dir / "enhanced",
        "augmented": stage2_dir / "augmented",
        "metrics": stage2_dir / "metrics",
        "comparison": stage2_dir / "comparison",
        "reports": stage2_dir / "reports"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tasks = []

    # 1. Discover Brain Dataset
    brain_dataset_dir = PROJECT_DIR.parent / "training_data_brain"
    if not brain_dataset_dir.exists():
        brain_dataset_dir = PROJECT_DIR / "training_data_brain"

    if brain_dataset_dir.exists():
        brats_root = list(brain_dataset_dir.glob("**/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"))
        if brats_root:
            p_folders = sorted([d for d in brats_root[0].iterdir() if d.is_dir()])
            if max_brain_patients:
                p_folders = p_folders[:max_brain_patients]

            for pf in p_folders:
                pid = pf.name
                for mod in ["t1", "t1ce", "t2", "flair"]:
                    nii_files = list(pf.glob(f"*{mod}.nii*"))
                    if nii_files:
                        tasks.append(("Brain", pid, mod.upper(), str(nii_files[0]), {k: str(v) for k, v in dirs.items()}))

    # 2. Discover Spine Dataset
    spine_dataset_dir = PROJECT_DIR.parent / "training_data_spine"
    if not spine_dataset_dir.exists():
        spine_dataset_dir = PROJECT_DIR / "training_data_spine"

    if spine_dataset_dir.exists():
        spine_files = sorted(list(spine_dataset_dir.glob("**/*.nii*")))
        for sfile in spine_files:
            pid = sfile.parent.name
            mod = sfile.name.split(".")[0]
            tasks.append(("Spine", pid, mod, str(sfile), {k: str(v) for k, v in dirs.items()}))

    print(f"\n[Stage 2] Processing {len(tasks)} MRI NIfTI volumes across Brain & Spine datasets...")

    num_workers = max(1, min(6, os.cpu_count() - 1))
    print(f" -> Parallel execution using {num_workers} CPU workers.")

    all_metrics = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_volume, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing MRI Volumes"):
            res = f.result()
            if res is not None:
                all_metrics.append(res)

    # 3. Export CSV Metrics
    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        csv_path = dirs["metrics"] / "stage2_quality_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n[Success] Preprocessing complete! Metrics saved to: {csv_path}")

        # Separate Brain & Spine CSVs
        brain_df = metrics_df[metrics_df["Dataset"] == "Brain"]
        spine_df = metrics_df[metrics_df["Dataset"] == "Spine"]
        brain_df.to_csv(dirs["metrics"] / "brain_preprocessing_metrics.csv", index=False)
        spine_df.to_csv(dirs["metrics"] / "spine_preprocessing_metrics.csv", index=False)

        # Generate Stage 2 Report
        report_path = dirs["reports"] / "stage2_preprocessing_report.md"
        Stage2ReportGenerator.generate_report(str(csv_path), str(report_path))

    print(f"\nTotal Stage 2 Pipeline Execution Time: {round(time.time() - start_time, 2)}s")


if __name__ == "__main__":
    max_b = 30
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        max_b = None
    run_stage2_pipeline(max_brain_patients=max_b)
