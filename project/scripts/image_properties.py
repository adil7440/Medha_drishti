import os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import laplace, sobel

class ImagePropertyExtractor:
    """
    Extracts geometric, spatial, statistical, texture, and signal quality metrics
    from NIfTI (.nii / .nii.gz) MRI volumes.
    """
    @staticmethod
    def compute_noise_estimate(data, bg_mask):
        """
        Estimates noise level using background standard deviation and Median Absolute Deviation (MAD).
        """
        bg_voxels = data[bg_mask]
        if len(bg_voxels) > 100:
            std_noise = float(np.std(bg_voxels))
            median_bg = np.median(bg_voxels)
            mad_noise = float(np.median(np.abs(bg_voxels - median_bg)) / 0.6745)
            return max(std_noise, mad_noise, 1e-5)
        else:
            # Fallback to low-intensity bottom 5 percentile voxels
            bottom_voxels = data[data < np.percentile(data, 5)]
            return max(float(np.std(bottom_voxels)), 1e-5)

    @staticmethod
    def compute_sharpness_and_edges(data_3d):
        """
        Computes 3D Laplacian variance (sharpness) and Sobel gradient magnitude (edge strength)
        on the central slice and 3D volume sample.
        """
        depth = data_3d.shape[2]
        mid_z = depth // 2
        # Slice range around middle
        start_z = max(0, mid_z - 5)
        end_z = min(depth, mid_z + 5)
        slice_stack = data_3d[:, :, start_z:end_z]

        # Sharpness via Laplacian variance
        lap_var_list = []
        sobel_grad_list = []

        for i in range(slice_stack.shape[2]):
            slc = slice_stack[:, :, i]
            if slc.max() == slc.min():
                continue
            # Normalize slice to [0, 1] for scale-invariant sharpness
            norm_slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            lap = laplace(norm_slc)
            lap_var_list.append(np.var(lap))

            # Sobel gradients
            gx = sobel(norm_slc, axis=0)
            gy = sobel(norm_slc, axis=1)
            mag = np.sqrt(gx**2 + gy**2)
            sobel_grad_list.append(np.mean(mag))

        avg_sharpness = float(np.mean(lap_var_list)) if lap_var_list else 0.0
        avg_edge_strength = float(np.mean(sobel_grad_list)) if sobel_grad_list else 0.0
        return avg_sharpness, avg_edge_strength

    @staticmethod
    def extract_properties(patient_id, modality, file_path):
        """
        Extracts comprehensive metadata and statistics for a single NIfTI file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        img = nib.load(str(path))
        header = img.header
        shape = img.shape
        zooms = header.get_zooms()[:3]
        dtype_str = str(header.get_data_dtype())
        
        try:
            orientation = ''.join(nib.aff2axcodes(img.affine))
        except Exception:
            orientation = "Unknown"

        qform_code = int(header['qform_code']) if 'qform_code' in header else 0
        sform_code = int(header['sform_code']) if 'sform_code' in header else 0
        affine_str = str(img.affine.round(2).tolist())

        data = img.get_fdata(dtype=np.float32)

        # Base Dictionary
        prop = {
            'Patient_ID': patient_id,
            'Modality': modality,
            'File_Name': path.name,
            'File_Path': str(path),
            'File_Size_MB': round(file_size_mb, 3),
            'Width': shape[0],
            'Height': shape[1],
            'Depth': shape[2] if len(shape) > 2 else 1,
            'Spacing_X': round(float(zooms[0]), 3),
            'Spacing_Y': round(float(zooms[1]), 3),
            'Spacing_Z': round(float(zooms[2]), 3) if len(zooms) > 2 else 1.0,
            'Voxel_Volume_mm3': round(float(zooms[0] * zooms[1] * (zooms[2] if len(zooms) > 2 else 1.0)), 3),
            'Data_Type': dtype_str,
            'Orientation': orientation,
            'QForm_Code': qform_code,
            'SForm_Code': sform_code,
            'Affine_Matrix': affine_str
        }

        # If Segmentation Mask
        if modality.lower() in ['seg', 'seg_mask', 'segmentation']:
            unique_labels, counts = np.unique(data, return_counts=True)
            label_dict = dict(zip(unique_labels.astype(int), counts))

            l0 = int(label_dict.get(0, 0))
            l1 = int(label_dict.get(1, 0))  # NCR/NET
            l2 = int(label_dict.get(2, 0))  # ED
            l4 = int(label_dict.get(4, 0))  # ET

            total_voxels = data.size
            tumor_voxels = l1 + l2 + l4
            voxel_vol = prop['Voxel_Volume_mm3']

            prop.update({
                'Min_Intensity': 0.0,
                'Max_Intensity': float(unique_labels.max()) if len(unique_labels) > 0 else 0.0,
                'Mean_Intensity': float(np.mean(data)),
                'Median_Intensity': float(np.median(data)),
                'Std_Intensity': float(np.std(data)),
                'Variance': float(np.var(data)),
                'Dynamic_Range': float(unique_labels.max()) if len(unique_labels) > 0 else 0.0,
                'Contrast': 0.0,
                'Entropy': 0.0,
                'Sharpness': 0.0,
                'Edge_Strength': 0.0,
                'Noise_Estimate': 0.0,
                'SNR': 0.0,
                'FG_BG_Ratio': round(tumor_voxels / max(l0, 1), 5),
                'Label_0_BG_Voxels': l0,
                'Label_1_NCR_Voxels': l1,
                'Label_2_ED_Voxels': l2,
                'Label_4_ET_Voxels': l4,
                'Total_Tumor_Voxels': tumor_voxels,
                'Tumor_Volume_cm3': round((tumor_voxels * voxel_vol) / 1000.0, 3),
                'NCR_Volume_cm3': round((l1 * voxel_vol) / 1000.0, 3),
                'ED_Volume_cm3': round((l2 * voxel_vol) / 1000.0, 3),
                'ET_Volume_cm3': round((l4 * voxel_vol) / 1000.0, 3),
                'Has_Tumor': tumor_voxels > 0
            })
            return prop

        # MRI Modality Intensity Statistics
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        dyn_range = max_val - min_val

        # Brain Foreground Mask (voxels > 0)
        fg_mask = data > 0
        bg_mask = ~fg_mask

        fg_count = np.sum(fg_mask)
        bg_count = np.sum(bg_mask)

        if fg_count > 0:
            fg_data = data[fg_mask]
            mean_intensity = float(np.mean(fg_data))
            median_intensity = float(np.median(fg_data))
            std_intensity = float(np.std(fg_data))
            variance = float(np.var(fg_data))

            # RMS Contrast = std / mean
            contrast = std_intensity / max(mean_intensity, 1e-5)

            # Shannon Entropy of foreground intensity distribution (256 bins)
            hist, _ = np.histogram(fg_data, bins=256, range=(fg_data.min(), fg_data.max()))
            hist_norm = hist / hist.sum()
            intensity_entropy = float(scipy_entropy(hist_norm + 1e-12, base=2))

            # Sharpness and Edge Strength
            sharpness, edge_strength = ImagePropertyExtractor.compute_sharpness_and_edges(data)

            # Noise Estimation
            noise_est = ImagePropertyExtractor.compute_noise_estimate(data, bg_mask)

            # SNR = mean_fg / noise
            snr = mean_intensity / max(noise_est, 1e-5)

            # Foreground-to-Background Ratio
            bg_mean = float(np.mean(data[bg_mask])) if bg_count > 0 else 0.01
            fg_bg_intensity_ratio = mean_intensity / max(bg_mean, 1e-5)

        else:
            mean_intensity = median_intensity = std_intensity = variance = 0.0
            contrast = intensity_entropy = sharpness = edge_strength = noise_est = snr = fg_bg_intensity_ratio = 0.0

        prop.update({
            'Min_Intensity': round(min_val, 2),
            'Max_Intensity': round(max_val, 2),
            'Mean_Intensity': round(mean_intensity, 2),
            'Median_Intensity': round(median_intensity, 2),
            'Std_Intensity': round(std_intensity, 2),
            'Variance': round(variance, 2),
            'Dynamic_Range': round(dyn_range, 2),
            'Contrast': round(contrast, 4),
            'Entropy': round(intensity_entropy, 4),
            'Sharpness': round(sharpness, 6),
            'Edge_Strength': round(edge_strength, 6),
            'Noise_Estimate': round(noise_est, 4),
            'SNR': round(snr, 4),
            'FG_BG_Ratio': round(fg_bg_intensity_ratio, 4),
            'Label_0_BG_Voxels': bg_count,
            'Label_1_NCR_Voxels': 0,
            'Label_2_ED_Voxels': 0,
            'Label_4_ET_Voxels': 0,
            'Total_Tumor_Voxels': 0,
            'Tumor_Volume_cm3': 0.0,
            'NCR_Volume_cm3': 0.0,
            'ED_Volume_cm3': 0.0,
            'ET_Volume_cm3': 0.0,
            'Has_Tumor': True  # BraTS patients are pathological cases
        })

        return prop
