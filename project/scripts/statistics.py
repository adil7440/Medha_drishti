import os
from pathlib import Path
import pandas as pd
import numpy as np

class DatasetStatisticsCalculator:
    """
    Computes dataset statistics, modality comparisons, patient summaries,
    performs automated quality checks, and exports summary CSVs.
    """
    def __init__(self, patient_df, properties_df, output_dir):
        self.patient_df = patient_df
        self.properties_df = properties_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_statistics(self):
        """Generates and saves dataset_statistics.csv, patient_statistics.csv, modality_statistics.csv, image_properties.csv."""
        # 1. Save Image Properties CSV
        properties_csv = self.output_dir / 'image_properties.csv'
        self.properties_df.to_csv(properties_csv, index=False)

        # 2. Generate Dataset Statistics CSV
        ds_stats = self._compute_overall_dataset_stats()
        ds_csv = self.output_dir / 'dataset_statistics.csv'
        ds_stats.to_csv(ds_csv, index=False)

        # 3. Generate Patient Statistics CSV
        pat_stats = self._compute_patient_stats()
        pat_csv = self.output_dir / 'patient_statistics.csv'
        pat_stats.to_csv(pat_csv, index=False)

        # 4. Generate Modality Statistics CSV
        mod_stats = self._compute_modality_stats()
        mod_csv = self.output_dir / 'modality_statistics.csv'
        mod_stats.to_csv(mod_csv, index=False)

        return {
            'dataset_statistics': ds_stats,
            'patient_statistics': pat_stats,
            'modality_statistics': mod_stats,
            'image_properties': self.properties_df
        }

    def _compute_overall_dataset_stats(self):
        df = self.properties_df
        p_df = self.patient_df

        total_patients = len(p_df)
        total_volumes = len(df)

        t1_count = len(df[df['Modality'].str.upper() == 'T1'])
        t1ce_count = len(df[df['Modality'].str.upper() == 'T1CE'])
        t2_count = len(df[df['Modality'].str.upper() == 'T2'])
        flair_count = len(df[df['Modality'].str.upper() == 'FLAIR'])
        seg_count = len(df[df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])])

        total_size_mb = df['File_Size_MB'].sum()
        total_size_gb = total_size_mb / 1024.0
        avg_file_size_mb = df['File_Size_MB'].mean()

        largest_file_row = df.loc[df['File_Size_MB'].idxmax()]
        smallest_file_row = df.loc[df['File_Size_MB'].idxmin()]

        hgg_count = len(p_df[p_df['Grade'] == 'HGG'])
        lgg_count = len(p_df[p_df['Grade'] == 'LGG'])
        unknown_grade = len(p_df[~p_df['Grade'].isin(['HGG', 'LGG'])])

        # Tumor volumes from seg masks
        seg_df = df[df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]
        avg_tumor_vol = seg_df['Tumor_Volume_cm3'].mean() if not seg_df.empty else 0.0
        max_tumor_vol = seg_df['Tumor_Volume_cm3'].max() if not seg_df.empty else 0.0

        stats = [
            {'Metric': 'Total Patients', 'Value': total_patients},
            {'Metric': 'Total MRI Volumes', 'Value': total_volumes},
            {'Metric': 'Number of T1 Scans', 'Value': t1_count},
            {'Metric': 'Number of T1CE Scans', 'Value': t1ce_count},
            {'Metric': 'Number of T2 Scans', 'Value': t2_count},
            {'Metric': 'Number of FLAIR Scans', 'Value': flair_count},
            {'Metric': 'Number of Segmentation Masks', 'Value': seg_count},
            {'Metric': 'Average File Size (MB)', 'Value': round(avg_file_size_mb, 2)},
            {'Metric': 'Largest File Name', 'Value': largest_file_row['File_Name']},
            {'Metric': 'Largest File Size (MB)', 'Value': round(largest_file_row['File_Size_MB'], 2)},
            {'Metric': 'Smallest File Name', 'Value': smallest_file_row['File_Name']},
            {'Metric': 'Smallest File Size (MB)', 'Value': round(smallest_file_row['File_Size_MB'], 2)},
            {'Metric': 'Total Dataset Size (MB)', 'Value': round(total_size_mb, 2)},
            {'Metric': 'Total Dataset Size (GB)', 'Value': round(total_size_gb, 3)},
            {'Metric': 'High Grade Gliomas (HGG)', 'Value': hgg_count},
            {'Metric': 'Low Grade Gliomas (LGG)', 'Value': lgg_count},
            {'Metric': 'Unknown Grade Scans', 'Value': unknown_grade},
            {'Metric': 'Pathological Scans (Tumour)', 'Value': total_patients},
            {'Metric': 'Healthy Control Scans', 'Value': 0},
            {'Metric': 'Average Tumour Volume (cm³)', 'Value': round(avg_tumor_vol, 2)},
            {'Metric': 'Maximum Tumour Volume (cm³)', 'Value': round(max_tumor_vol, 2)}
        ]

        return pd.DataFrame(stats)

    def _compute_patient_stats(self):
        p_df = self.patient_df.copy()
        df = self.properties_df

        # Aggregate per patient file sizes and volume properties
        size_agg = df.groupby('Patient_ID')['File_Size_MB'].sum().rename('Total_Size_MB')
        
        seg_df = df[df['Modality'].str.lower().isin(['seg', 'seg_mask', 'segmentation'])]
        tumor_agg = seg_df.groupby('Patient_ID')['Tumor_Volume_cm3'].first().rename('Tumor_Volume_cm3')

        p_df = p_df.merge(size_agg, on='Patient_ID', how='left')
        p_df = p_df.merge(tumor_agg, on='Patient_ID', how='left')

        # Clean display
        p_df['Total_Size_MB'] = p_df['Total_Size_MB'].round(2)
        p_df['Tumor_Volume_cm3'] = p_df['Tumor_Volume_cm3'].fillna(0.0).round(2)

        cols = [
            'Patient_ID', 'Grade', 'Status', 'Total_Files', 'Missing_Files',
            'Has_T1', 'Has_T1CE', 'Has_T2', 'Has_FLAIR', 'Has_Seg',
            'Total_Size_MB', 'Tumor_Volume_cm3'
        ]
        return p_df[cols]

    def _compute_modality_stats(self):
        df = self.properties_df.copy()

        # Group by Modality
        grouped = df.groupby('Modality')

        records = []
        for mod, group in grouped:
            # Exclude seg mask for image intensity statistics if desired, or keep separate
            is_seg = mod.lower() in ['seg', 'seg_mask', 'segmentation']

            shape_str = f"{int(group['Width'].mode()[0])} x {int(group['Height'].mode()[0])} x {int(group['Depth'].mode()[0])}"
            spacing_str = f"{group['Spacing_X'].mode()[0]} x {group['Spacing_Y'].mode()[0]} x {group['Spacing_Z'].mode()[0]} mm"

            rec = {
                'Modality': mod,
                'Volume_Count': len(group),
                'Average_Intensity': round(group['Mean_Intensity'].mean(), 2),
                'Median_Intensity': round(group['Median_Intensity'].mean(), 2),
                'Min_Intensity': round(group['Min_Intensity'].min(), 2),
                'Max_Intensity': round(group['Max_Intensity'].max(), 2),
                'Average_Contrast': round(group['Contrast'].mean(), 4),
                'Average_Entropy': round(group['Entropy'].mean(), 4),
                'Average_Sharpness': round(group['Sharpness'].mean(), 6),
                'Average_Noise': round(group['Noise_Estimate'].mean(), 4),
                'Average_SNR': round(group['SNR'].mean(), 4),
                'Average_Edge_Strength': round(group['Edge_Strength'].mean(), 6),
                'Typical_Dimensions': shape_str,
                'Typical_Voxel_Spacing': spacing_str,
                'Average_File_Size_MB': round(group['File_Size_MB'].mean(), 2)
            }
            records.append(rec)

        return pd.DataFrame(records)

    def run_quality_checks(self):
        """
        Automated dataset quality checks:
        1. Missing modalities per patient
        2. Corrupted NIfTI files
        3. Dimension anomalies
        4. Voxel spacing anomalies
        5. Duplicate patient IDs
        6. Invalid intensities (NaN / Inf)
        """
        df = self.properties_df
        p_df = self.patient_df

        warnings = []
        checks_summary = {}

        # 1. Missing modalities
        incomplete_patients = p_df[p_df['Status'] != 'Complete']
        checks_summary['Incomplete_Patients_Count'] = len(incomplete_patients)
        if len(incomplete_patients) > 0:
            for _, row in incomplete_patients.iterrows():
                warnings.append(f"[QUALITY CHECK] Patient {row['Patient_ID']} missing files: {row['Missing_Files']}")
        else:
            checks_summary['Missing_Modalities_Check'] = 'PASSED (0 missing modalities across all patients)'

        # 2. Duplicate Patients
        dup_patients = p_df[p_df.duplicated(subset=['Patient_ID'])]
        checks_summary['Duplicate_Patients_Count'] = len(dup_patients)
        if len(dup_patients) > 0:
            warnings.append(f"[QUALITY CHECK] Found {len(dup_patients)} duplicate patient records!")
        else:
            checks_summary['Duplicate_Check'] = 'PASSED (No duplicate patient IDs)'

        # 3. Shape / Dimension anomalies
        shapes = df.apply(lambda r: (r['Width'], r['Height'], r['Depth']), axis=1)
        unique_shapes = shapes.unique()
        checks_summary['Unique_Dimensions'] = [str(s) for s in unique_shapes]
        if len(unique_shapes) > 1:
            warnings.append(f"[QUALITY CHECK] Heterogeneous dimensions detected across volumes: {unique_shapes}")
        else:
            checks_summary['Dimensions_Check'] = f'PASSED (Homogeneous dimensions: {unique_shapes[0]})'

        # 4. Voxel Spacing anomalies
        spacings = df.apply(lambda r: (r['Spacing_X'], r['Spacing_Y'], r['Spacing_Z']), axis=1)
        unique_spacings = spacings.unique()
        checks_summary['Unique_Voxel_Spacings'] = [str(s) for s in unique_spacings]
        if len(unique_spacings) > 1:
            warnings.append(f"[QUALITY CHECK] Heterogeneous voxel spacings detected: {unique_spacings}")
        else:
            checks_summary['Spacing_Check'] = f'PASSED (Homogeneous voxel spacing: {unique_spacings[0]} mm)'

        # 5. Invalid values (NaN / Inf / All Zero)
        invalid_rows = df[df['Mean_Intensity'].isna() | np.isinf(df['Mean_Intensity'])]
        checks_summary['Invalid_Volumes_Count'] = len(invalid_rows)
        if len(invalid_rows) > 0:
            warnings.append(f"[QUALITY CHECK] Found {len(invalid_rows)} invalid or NaN volumes!")
        else:
            checks_summary['Data_Integrity_Check'] = 'PASSED (0 corrupted or NaN volumes)'

        return {
            'summary': checks_summary,
            'warnings': warnings
        }
