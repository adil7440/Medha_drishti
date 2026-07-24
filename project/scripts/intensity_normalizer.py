import numpy as np


class IntensityNormalizer:
    """
    Provides intensity normalization techniques for MRI volumes:
    - Min-Max Normalization
    - Z-Score Normalization (Standardization)
    - Percentile Clipping
    """

    @staticmethod
    def percentile_clipping(data: np.ndarray, p_min: float = 0.5, p_max: float = 99.5) -> np.ndarray:
        """
        Clips voxel intensities to specified lower and upper percentiles to remove extreme outliers.
        """
        data = data.astype(np.float32)
        non_zero_mask = data > 0
        if not np.any(non_zero_mask):
            return data

        lower_bound = np.percentile(data[non_zero_mask], p_min)
        upper_bound = np.percentile(data[non_zero_mask], p_max)
        
        clipped = np.clip(data, lower_bound, upper_bound)
        return clipped

    @staticmethod
    def min_max(data: np.ndarray, out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
        """
        Scales intensity values linearly to specified range [out_min, out_max].
        """
        data = data.astype(np.float32)
        dmin, dmax = np.min(data), np.max(data)
        if dmax == dmin:
            return np.zeros_like(data, dtype=np.float32)
        
        scaled = (data - dmin) / (dmax - dmin)
        return scaled * (out_max - out_min) + out_min

    @staticmethod
    def z_score(data: np.ndarray, mask_non_zero: bool = True) -> np.ndarray:
        """
        Performs Z-score standardization (zero mean, unit variance).
        """
        data = data.astype(np.float32)
        if mask_non_zero:
            mask = data > 0
            if not np.any(mask):
                return data
            mean = np.mean(data[mask])
            std = np.std(data[mask])
            if std == 0:
                std = 1e-8
            out = np.zeros_like(data, dtype=np.float32)
            out[mask] = (data[mask] - mean) / std
            return out
        else:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                std = 1e-8
            return (data - mean) / std
