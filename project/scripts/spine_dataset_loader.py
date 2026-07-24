import os
import re
from pathlib import Path
import pandas as pd


class SpineDatasetLoader:
    """
    Scans the spine MRI training dataset directory, discovers patient folders,
    classifies them as Normal or Pathological, detects MRI modalities from
    filenames, checks completeness, and builds a patient inventory DataFrame.
    """

    MODALITY_KEYWORDS = {
        'T1W': ['eT1W_TSE_CLEAR', 'eT1W_TSE', 'T1W_TSE_sag',
                 'T1W_TSE_GADO_PRE', 'T1W_TSE_GADO_POST', 'T1W_TSE_POST',
                 'MobiView_eT1W_TSE'],
        'T2W': ['eT2W_TSE_DRIVE_HR', 'eT2W_TSE_CLEAR', 'eT2W_TSE',
                 'SC_MobiView_eT2W_TSE', 'eeT2_SPAIR_cor'],
        'STIR': ['eeSTIR_TSE_COR', 'eSTIR_TSE', 'eSTIR_longTE',
                 'STIR_TSE', 'SC_MobiView_eSTIR_TSE'],
        'Survey': ['MobiView_SURVEY_SAG'],
    }

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def _classify_patient(self, patient_dir_name):
        """Returns 'Normal' or 'Pathological' based on parent directory name."""
        return None

    def _detect_modalities(self, files):
        """Detect which MRI modality categories are present from filenames."""
        detected = set()
        for f in files:
            fname_upper = f.name.upper()
            for mod_key, keywords in self.MODALITY_KEYWORDS.items():
                for kw in keywords:
                    if kw.upper() in fname_upper:
                        detected.add(mod_key)
                        break
        return sorted(detected)

    def scan_dataset(self):
        """
        Scans Normal and Pathological subdirectories.
        Returns a pandas DataFrame with patient discovery information.
        """
        categories = {
            'Normal': self.base_dir / 'Normal Spine MRI Datasets',
            'Pathological': self.base_dir / 'Pathological Spine MRI Datasets',
        }

        records = []
        for class_label, cat_dir in categories.items():
            if not cat_dir.exists():
                print(f"[Warning] Directory not found: {cat_dir}")
                continue

            for patient_dir in sorted(cat_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue

                nii_files = list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.nii"))
                if not nii_files:
                    continue

                patient_id = patient_dir.name
                modalities = self._detect_modalities(nii_files)
                num_files = len(nii_files)

                total_size_mb = sum(f.stat().st_size for f in nii_files) / (1024 * 1024)

                has_t1w = any('T1W' in m for m in modalities)
                has_t2w = any('T2W' in m for m in modalities)
                has_stir = 'STIR' in modalities
                has_survey = 'Survey' in modalities

                missing = []
                if not has_t1w:
                    missing.append('T1W')
                if not has_t2w:
                    missing.append('T2W')

                status = 'Complete' if not missing else f'Incomplete (Missing: {", ".join(missing)})'

                record = {
                    'Patient_ID': patient_id,
                    'Patient_Dir': str(patient_dir),
                    'Class': class_label,
                    'Modalities_Detected': ", ".join(modalities),
                    'Has_T1W': has_t1w,
                    'Has_T2W': has_t2w,
                    'Has_STIR': has_stir,
                    'Has_Survey': has_survey,
                    'Total_Files': num_files,
                    'Total_Size_MB': round(total_size_mb, 2),
                    'Missing_Core_Modalities': ", ".join(missing) if missing else 'None',
                    'Status': status,
                }
                records.append(record)

        return pd.DataFrame(records)
