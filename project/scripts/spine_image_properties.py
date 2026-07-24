from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import laplace, sobel


class SpineImagePropertyExtractor:
    """
    Extracts geometric, spatial, statistical, texture, and signal quality metrics
    from NIfTI (.nii.gz) Spine MRI volumes.
    """

    @staticmethod
    def _classify_modality(filename):
        """Classify a filename into a broad modality category."""
        fname = filename.upper()
        if any(k in fname for k in ['T1W_TSE_GADO', 'T1W_TSE_POST']):
            return 'T1W_GADO'
        if 'T1W' in fname or 'ET1W' in fname:
            return 'T1W'
        if 'T2W' in fname or 'ET2W' in fname:
            return 'T2W'
        if 'STIR' in fname:
            return 'STIR'
        if 'SURVEY' in fname:
            return 'Survey'
        if 'SPAIR' in fname:
            return 'SPAIR'
        return 'Other'

    @staticmethod
    def compute_noise_estimate(data, bg_mask):
        """Estimates noise level using background standard deviation and MAD."""
        bg_voxels = data[bg_mask]
        if len(bg_voxels) > 100:
            std_noise = float(np.std(bg_voxels))
            median_bg = np.median(bg_voxels)
            mad_noise = float(np.median(np.abs(bg_voxels - median_bg)) / 0.6745)
            return max(std_noise, mad_noise, 1e-5)
        else:
            bottom_voxels = data[data < np.percentile(data, 5)]
            return max(float(np.std(bottom_voxels)), 1e-5)

    @staticmethod
    def compute_sharpness_and_edges(data_3d):
        """Computes 3D Laplacian variance (sharpness) and Sobel gradient magnitude."""
        depth = data_3d.shape[2]
        mid_z = depth // 2
        start_z = max(0, mid_z - 5)
        end_z = min(depth, mid_z + 5)
        slice_stack = data_3d[:, :, start_z:end_z]

        lap_var_list = []
        sobel_grad_list = []

        for i in range(slice_stack.shape[2]):
            slc = slice_stack[:, :, i]
            if slc.max() == slc.min():
                continue
            norm_slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            lap = laplace(norm_slc)
            lap_var_list.append(np.var(lap))

            gx = sobel(norm_slc, axis=0)
            gy = sobel(norm_slc, axis=1)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            sobel_grad_list.append(np.mean(mag))

        avg_sharpness = float(np.mean(lap_var_list)) if lap_var_list else 0.0
        avg_edge_strength = float(np.mean(sobel_grad_list)) if sobel_grad_list else 0.0
        return avg_sharpness, avg_edge_strength

    @staticmethod
    def extract_properties(patient_id, file_path):
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

        modality = SpineImagePropertyExtractor._classify_modality(path.name)

        # Intensity statistics
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        dyn_range = max_val - min_val

        # Foreground mask (non-zero voxels typical for MRI signal)
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

            contrast = std_intensity / max(mean_intensity, 1e-5)

            hist, _ = np.histogram(fg_data, bins=256,
                                   range=(fg_data.min(), fg_data.max()))
            hist_norm = hist / hist.sum()
            intensity_entropy = float(scipy_entropy(hist_norm + 1e-12, base=2))

            sharpness, edge_strength = SpineImagePropertyExtractor.compute_sharpness_and_edges(data)

            noise_est = SpineImagePropertyExtractor.compute_noise_estimate(data, bg_mask)

            snr = mean_intensity / max(noise_est, 1e-5)

            bg_mean = float(np.mean(data[bg_mask])) if bg_count > 0 else 0.01
            fg_bg_intensity_ratio = mean_intensity / max(bg_mean, 1e-5)
        else:
            mean_intensity = median_intensity = std_intensity = variance = 0.0
            contrast = intensity_entropy = sharpness = edge_strength = 0.0
            noise_est = snr = fg_bg_intensity_ratio = 0.0

        prop = {
            'Patient_ID': patient_id,
            'Modality_Category': modality,
            'File_Name': path.name,
            'File_Path': str(path),
            'File_Size_MB': round(file_size_mb, 3),
            'Width': shape[0],
            'Height': shape[1],
            'Depth': shape[2] if len(shape) > 2 else 1,
            'Spacing_X': round(float(zooms[0]), 3),
            'Spacing_Y': round(float(zooms[1]), 3),
            'Spacing_Z': round(float(zooms[2]), 3) if len(zooms) > 2 else 1.0,
            'Voxel_Volume_mm3': round(float(
                zooms[0] * zooms[1] * (zooms[2] if len(zooms) > 2 else 1.0)), 3),
            'Data_Type': dtype_str,
            'Orientation': orientation,
            'QForm_Code': qform_code,
            'SForm_Code': sform_code,
            'Affine_Matrix': affine_str,
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
            'FG_Voxel_Count': int(fg_count),
            'BG_Voxel_Count': int(bg_count),
        }
        return prop
