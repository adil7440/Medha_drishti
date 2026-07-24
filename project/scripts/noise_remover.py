import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma


class NoiseRemover:
    """
    Implements and compares noise reduction filters for 2D/3D MRI volumes:
    - Gaussian Filter
    - Median Filter
    - Bilateral Filter
    - Non-Local Means (NLM) Denoising
    """

    @staticmethod
    def gaussian_denoise(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Applies Gaussian smoothing across the volume."""
        return gaussian_filter(data.astype(np.float32), sigma=sigma)

    @staticmethod
    def median_denoise(data: np.ndarray, size: int = 3) -> np.ndarray:
        """Applies median filtering to suppress salt-and-pepper noise."""
        return median_filter(data.astype(np.float32), size=size)

    @staticmethod
    def bilateral_denoise_slice(slice_2d: np.ndarray, d: int = 5, sigma_color: float = 25.0, sigma_space: float = 25.0) -> np.ndarray:
        """Applies 2D Bilateral filtering preserving sharp anatomical edges."""
        # Normalize to uint8 for opencv bilateral filter
        s_min, s_max = np.min(slice_2d), np.max(slice_2d)
        if s_max == s_min:
            return slice_2d
        
        norm_slice = ((slice_2d - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        filtered = cv2.bilateralFilter(norm_slice, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return (filtered.astype(np.float32) / 255.0) * (s_max - s_min) + s_min

    @classmethod
    def bilateral_denoise_volume(cls, data_3d: np.ndarray, d: int = 5, sigma_color: float = 25.0, sigma_space: float = 25.0) -> np.ndarray:
        """Applies bilateral filtering slice-wise across the 3D volume."""
        output = np.zeros_like(data_3d, dtype=np.float32)
        for z in range(data_3d.shape[2]):
            output[:, :, z] = cls.bilateral_denoise_slice(data_3d[:, :, z], d=d, sigma_color=sigma_color, sigma_space=sigma_space)
        return output

    @staticmethod
    def nlm_denoise_slice(slice_2d: np.ndarray, h_factor: float = 0.8) -> np.ndarray:
        """Applies Non-Local Means (NLM) Denoising on a 2D slice."""
        data_norm = slice_2d.astype(np.float32)
        s_min, s_max = np.min(data_norm), np.max(data_norm)
        if s_max == s_min:
            return slice_2d

        norm = (data_norm - s_min) / (s_max - s_min + 1e-8)
        sigma_est = np.mean(estimate_sigma(norm))
        if sigma_est == 0:
            sigma_est = 0.05
            
        denoised = denoise_nl_means(
            norm,
            h=h_factor * sigma_est,
            fast_mode=True,
            patch_size=5,
            patch_distance=7
        )
        return denoised * (s_max - s_min) + s_min

    @classmethod
    def nlm_denoise_volume(cls, data_3d: np.ndarray, h_factor: float = 0.8) -> np.ndarray:
        """Applies NLM Denoising slice-wise across the 3D volume."""
        output = np.zeros_like(data_3d, dtype=np.float32)
        for z in range(data_3d.shape[2]):
            output[:, :, z] = cls.nlm_denoise_slice(data_3d[:, :, z], h_factor=h_factor)
        return output
