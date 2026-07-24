import os
from pathlib import Path
import pandas as pd

class BrainDatasetLoader:
    """
    Scans the training dataset directory, discovers patient folders,
    detects MRI modalities (T1, T1CE, T2, FLAIR, SEG), checks completeness,
    and loads dataset grade metadata.
    """
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.mapping_df = None
        self.patient_records = []
        self._load_name_mapping()

    def _load_name_mapping(self):
        """Attempts to find name_mapping.csv inside the training directory to load HGG/LGG grades."""
        mapping_files = list(self.base_dir.rglob("name_mapping.csv"))
        if mapping_files:
            try:
                df = pd.read_csv(mapping_files[0])
                if 'BraTS_2020_subject_ID' in df.columns and 'Grade' in df.columns:
                    self.mapping_df = df.set_index('BraTS_2020_subject_ID')
                elif 'BraTS20_ID' in df.columns and 'Grade' in df.columns:
                    self.mapping_df = df.set_index('BraTS20_ID')
            except Exception as e:
                print(f"[Warning] Could not parse name_mapping.csv: {e}")

    def scan_dataset(self):
        """
        Scans all patient subdirectories in self.base_dir.
        Returns a pandas DataFrame containing patient discovery information.
        """
        patient_dirs = set()
        for root, dirs, files in os.walk(self.base_dir):
            nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
            if nii_files:
                patient_dirs.add(Path(root))

        patient_dirs = sorted(list(patient_dirs))
        records = []

        for p_dir in patient_dirs:
            patient_id = p_dir.name
            files = list(p_dir.glob("*.nii")) + list(p_dir.glob("*.nii.gz"))
            if not files:
                continue

            # Detect modalities
            t1_file = None
            t1ce_file = None
            t2_file = None
            flair_file = None
            seg_file = None

            for f in files:
                fname = f.name.lower()
                if '_t1ce' in fname or 't1ce.' in fname:
                    t1ce_file = f
                elif '_t1' in fname or 't1.' in fname:
                    t1_file = f
                elif '_t2' in fname or 't2.' in fname:
                    t2_file = f
                elif '_flair' in fname or 'flair.' in fname:
                    flair_file = f
                elif '_seg' in fname or 'seg.' in fname:
                    seg_file = f

            modalities_found = {
                'T1': t1_file is not None,
                'T1CE': t1ce_file is not None,
                'T2': t2_file is not None,
                'FLAIR': flair_file is not None,
                'Seg_Mask': seg_file is not None
            }

            missing = [mod for mod, found in modalities_found.items() if not found]
            num_files = len(files)
            status = 'Complete' if len(missing) == 0 else f'Incomplete (Missing: {", ".join(missing)})'

            grade = 'Unknown'
            if self.mapping_df is not None and patient_id in self.mapping_df.index:
                grade = self.mapping_df.loc[patient_id, 'Grade']

            record = {
                'Patient_ID': patient_id,
                'Patient_Dir': str(p_dir),
                'Grade': grade,
                'T1_Path': str(t1_file) if t1_file else None,
                'T1CE_Path': str(t1ce_file) if t1ce_file else None,
                'T2_Path': str(t2_file) if t2_file else None,
                'FLAIR_Path': str(flair_file) if flair_file else None,
                'Seg_Path': str(seg_file) if seg_file else None,
                'Has_T1': modalities_found['T1'],
                'Has_T1CE': modalities_found['T1CE'],
                'Has_T2': modalities_found['T2'],
                'Has_FLAIR': modalities_found['FLAIR'],
                'Has_Seg': modalities_found['Seg_Mask'],
                'Total_Files': num_files,
                'Missing_Files': ", ".join(missing) if missing else 'None',
                'Status': status
            }
            records.append(record)

        self.patient_records = records
        return pd.DataFrame(records)
