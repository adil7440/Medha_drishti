from pathlib import Path
import pandas as pd
import numpy as np


class SpineDatasetStatisticsCalculator:
    """
    Computes dataset statistics, modality comparisons, patient summaries,
    performs automated quality checks, and exports summary CSVs
    for the Spine MRI dataset.
    """

    def __init__(self, patient_df, properties_df, output_dir):
        self.patient_df = patient_df
        self.properties_df = properties_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_statistics(self):
        """Generates and saves all statistics CSVs."""
        properties_csv = self.output_dir / 'spine_image_properties.csv'
        self.properties_df.to_csv(properties_csv, index=False)

        ds_stats = self._compute_overall_dataset_stats()
        ds_csv = self.output_dir / 'spine_dataset_statistics.csv'
        ds_stats.to_csv(ds_csv, index=False)

        pat_stats = self._compute_patient_stats()
        pat_csv = self.output_dir / 'spine_patient_statistics.csv'
        pat_stats.to_csv(pat_csv, index=False)

        mod_stats = self._compute_modality_stats()
        mod_csv = self.output_dir / 'spine_modality_statistics.csv'
        mod_stats.to_csv(mod_csv, index=False)

        class_stats = self._compute_class_comparison()
        class_csv = self.output_dir / 'spine_class_comparison.csv'
        class_stats.to_csv(class_csv, index=False)

        return {
            'dataset_statistics': ds_stats,
            'patient_statistics': pat_stats,
            'modality_statistics': mod_stats,
            'class_comparison': class_stats,
            'image_properties': self.properties_df,
        }

    def _compute_overall_dataset_stats(self):
        df = self.properties_df
        p_df = self.patient_df

        total_patients = len(p_df)
        total_volumes = len(df)
        normal_count = len(p_df[p_df['Class'] == 'Normal'])
        pathological_count = len(p_df[p_df['Class'] == 'Pathological'])

        t1w_count = len(df[df['Modality_Category'] == 'T1W'])
        t1w_gado_count = len(df[df['Modality_Category'] == 'T1W_GADO'])
        t2w_count = len(df[df['Modality_Category'] == 'T2W'])
        stir_count = len(df[df['Modality_Category'] == 'STIR'])
        survey_count = len(df[df['Modality_Category'] == 'Survey'])
        other_count = len(df[df['Modality_Category'].isin(['SPAIR', 'Other'])])

        total_size_mb = df['File_Size_MB'].sum()
        total_size_gb = total_size_mb / 1024.0
        avg_file_size_mb = df['File_Size_MB'].mean()

        largest_file_row = df.loc[df['File_Size_MB'].idxmax()]
        smallest_file_row = df.loc[df['File_Size_MB'].idxmin()]

        img_df = df[~df['Modality_Category'].isin(['Survey', 'Other'])]
        avg_snr = img_df['SNR'].mean() if not img_df.empty else 0.0
        avg_contrast = img_df['Contrast'].mean() if not img_df.empty else 0.0
        avg_entropy = img_df['Entropy'].mean() if not img_df.empty else 0.0

        stats = [
            {'Metric': 'Total Patients', 'Value': total_patients},
            {'Metric': 'Total Normal Patients', 'Value': normal_count},
            {'Metric': 'Total Pathological Patients', 'Value': pathological_count},
            {'Metric': 'Total MRI Volumes', 'Value': total_volumes},
            {'Metric': 'Number of T1W Scans', 'Value': t1w_count},
            {'Metric': 'Number of T1W GADO Scans', 'Value': t1w_gado_count},
            {'Metric': 'Number of T2W Scans', 'Value': t2w_count},
            {'Metric': 'Number of STIR Scans', 'Value': stir_count},
            {'Metric': 'Number of Survey/Locator Scans', 'Value': survey_count},
            {'Metric': 'Other/Special Sequences', 'Value': other_count},
            {'Metric': 'Average File Size (MB)', 'Value': round(avg_file_size_mb, 2)},
            {'Metric': 'Largest File Name', 'Value': largest_file_row['File_Name']},
            {'Metric': 'Largest File Size (MB)', 'Value': round(largest_file_row['File_Size_MB'], 2)},
            {'Metric': 'Smallest File Name', 'Value': smallest_file_row['File_Name']},
            {'Metric': 'Smallest File Size (MB)', 'Value': round(smallest_file_row['File_Size_MB'], 2)},
            {'Metric': 'Total Dataset Size (MB)', 'Value': round(total_size_mb, 2)},
            {'Metric': 'Total Dataset Size (GB)', 'Value': round(total_size_gb, 3)},
            {'Metric': 'Average SNR (Excl. Survey)', 'Value': round(avg_snr, 4)},
            {'Metric': 'Average RMS Contrast (Excl. Survey)', 'Value': round(avg_contrast, 4)},
            {'Metric': 'Average Entropy (Excl. Survey)', 'Value': round(avg_entropy, 4)},
        ]

        return pd.DataFrame(stats)

    def _compute_patient_stats(self):
        p_df = self.patient_df.copy()
        df = self.properties_df

        size_agg = df.groupby('Patient_ID')['File_Size_MB'].sum().rename('Props_Total_Size_MB')
        vol_count = df.groupby('Patient_ID').size().rename('Volume_Count')

        p_df = p_df.merge(size_agg, on='Patient_ID', how='left')
        p_df = p_df.merge(vol_count, on='Patient_ID', how='left')

        p_df['Computed_Size_MB'] = p_df['Props_Total_Size_MB'].round(2)

        cols = [
            'Patient_ID', 'Class', 'Status', 'Total_Files', 'Volume_Count',
            'Has_T1W', 'Has_T2W', 'Has_STIR', 'Has_Survey',
            'Total_Size_MB', 'Computed_Size_MB', 'Modalities_Detected'
        ]
        return p_df[cols]

    def _compute_modality_stats(self):
        df = self.properties_df.copy()
        grouped = df.groupby('Modality_Category')

        records = []
        for mod, group in grouped:
            shape_str = (f"{int(group['Width'].mode()[0])} x "
                         f"{int(group['Height'].mode()[0])} x "
                         f"{int(group['Depth'].mode()[0])}")
            spacing_str = (f"{group['Spacing_X'].mode()[0]} x "
                           f"{group['Spacing_Y'].mode()[0]} x "
                           f"{group['Spacing_Z'].mode()[0]} mm")

            rec = {
                'Modality': mod,
                'Volume_Count': len(group),
                'Patients_With_Modality': group['Patient_ID'].nunique(),
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
                'Average_File_Size_MB': round(group['File_Size_MB'].mean(), 2),
            }
            records.append(rec)

        return pd.DataFrame(records)

    def _compute_class_comparison(self):
        df = self.properties_df.copy()
        p_df = self.patient_df

        img_df = df[~df['Modality_Category'].isin(['Survey', 'Other'])]

        normal_pids = p_df[p_df['Class'] == 'Normal']['Patient_ID'].tolist()
        path_pids = p_df[p_df['Class'] == 'Pathological']['Patient_ID'].tolist()

        normal_imgs = img_df[img_df['Patient_ID'].isin(normal_pids)]
        path_imgs = img_df[img_df['Patient_ID'].isin(path_pids)]

        def _stats_for(subset, label):
            if subset.empty:
                return {}
            return {
                'Class': label,
                'Patient_Count': len(set(subset['Patient_ID'])),
                'Volume_Count': len(subset),
                'Mean_Intensity_Avg': round(subset['Mean_Intensity'].mean(), 2),
                'Contrast_Avg': round(subset['Contrast'].mean(), 4),
                'Entropy_Avg': round(subset['Entropy'].mean(), 4),
                'Sharpness_Avg': round(subset['Sharpness'].mean(), 6),
                'SNR_Avg': round(subset['SNR'].mean(), 4),
                'Noise_Avg': round(subset['Noise_Estimate'].mean(), 4),
                'Edge_Strength_Avg': round(subset['Edge_Strength'].mean(), 6),
                'File_Size_MB_Avg': round(subset['File_Size_MB'].mean(), 2),
                'Dimensions_Variants': subset.groupby(
                    ['Width', 'Height', 'Depth']).ngroups,
            }

        normal_stats = _stats_for(normal_imgs, 'Normal')
        path_stats = _stats_for(path_imgs, 'Pathological')

        records = [r for r in [normal_stats, path_stats] if r]
        return pd.DataFrame(records)

    def run_quality_checks(self):
        """Automated dataset quality checks for spine MRI."""
        df = self.properties_df
        p_df = self.patient_df

        warnings = []
        checks_summary = {}

        # 1. Missing core modalities
        incomplete = p_df[p_df['Status'] != 'Complete']
        checks_summary['Incomplete_Patients_Count'] = len(incomplete)
        if len(incomplete) > 0:
            for _, row in incomplete.iterrows():
                warnings.append(
                    f"[QUALITY] Patient {row['Patient_ID']} missing: "
                    f"{row['Missing_Core_Modalities']}")
        else:
            checks_summary['Core_Modalities_Check'] = (
                'PASSED (All patients have T1W and T2W)')

        # 2. Duplicate Patients
        dup = p_df[p_df.duplicated(subset=['Patient_ID'])]
        checks_summary['Duplicate_Patients_Count'] = len(dup)
        if len(dup) > 0:
            warnings.append(f"[QUALITY] Found {len(dup)} duplicate patient records!")
        else:
            checks_summary['Duplicate_Check'] = 'PASSED (No duplicate patient IDs)'

        # 3. Dimension anomalies
        shapes = df.apply(lambda r: (r['Width'], r['Height'], r['Depth']), axis=1)
        unique_shapes = shapes.unique()
        checks_summary['Unique_Dimensions'] = [str(s) for s in unique_shapes]
        if len(unique_shapes) > 3:
            warnings.append(
                f"[QUALITY] Many heterogeneous dimensions detected: "
                f"{len(unique_shapes)} unique shapes")
        else:
            checks_summary['Dimensions_Check'] = (
                f'PASSED ({len(unique_shapes)} unique dimension variants)')

        # 4. Voxel Spacing anomalies
        spacings = df.apply(
            lambda r: (r['Spacing_X'], r['Spacing_Y'], r['Spacing_Z']), axis=1)
        unique_spacings = spacings.unique()
        checks_summary['Unique_Voxel_Spacings'] = [str(s) for s in unique_spacings]

        # 5. Invalid values
        invalid = df[df['Mean_Intensity'].isna() | np.isinf(df['Mean_Intensity'])]
        checks_summary['Invalid_Volumes_Count'] = len(invalid)
        if len(invalid) > 0:
            warnings.append(f"[QUALITY] Found {len(invalid)} invalid/NaN volumes!")
        else:
            checks_summary['Data_Integrity_Check'] = (
                'PASSED (0 corrupted or NaN volumes)')

        # 6. All-zero volumes
        zero_vols = df[df['Max_Intensity'] == 0]
        checks_summary['AllZero_Volumes_Count'] = len(zero_vols)
        if len(zero_vols) > 0:
            warnings.append(
                f"[QUALITY] Found {len(zero_vols)} all-zero volumes!")

        return {
            'summary': checks_summary,
            'warnings': warnings,
        }
