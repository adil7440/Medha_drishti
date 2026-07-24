import numpy as np
import cv2
from skimage import exposure


class ContrastEnhancer:
    """
    Implements MRI contrast enhancement algorithms:
    - Histogram Equalization (HE)
    - Adaptive Histogram Equalization (AHE)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """

    @staticmethod
    def histogram_equalization_slice(slice_2d: np.ndarray) -> np.ndarray:
        """Applies global histogram equalization on a 2D slice."""
        s_min, s_max = np.min(slice_2d), np.max(slice_2d)
        if s_max == s_min:
            return slice_2d

        norm = ((slice_2d - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        equalized = cv2.equalizeHist(norm)
        return (equalized.astype(np.float32) / 255.0) * (s_max - s_min) + s_min

    @staticmethod
    def clahe_slice(slice_2d: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on a 2D slice."""
        s_min, s_max = np.min(slice_2d), np.max(slice_2d)
        if s_max == s_min:
            return slice_2d

        norm = ((slice_2d - s_min) / (s_max - s_min + 1e-8) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(norm)
        return (enhanced.astype(np.float32) / 255.0) * (s_max - s_min) + s_min

    @staticmethod
    def ahe_slice(slice_2d: np.ndarray, kernel_size: int = 16) -> np.ndarray:
        """Applies Adaptive Histogram Equalization (AHE) using skimage exposure.equalize_adapthist."""
        s_min, s_max = np.min(slice_2d), np.max(slice_2d)
        if s_max == s_min:
            return slice_2d

        norm = (slice_2d - s_min) / (s_max - s_min + 1e-8)
        enhanced = exposure.equalize_adapthist(norm, kernel_size=kernel_size, clip_limit=0.03)
        return enhanced * (s_max - s_min) + s_min

    @classmethod
    def apply_clahe_volume(cls, data_3d: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """Applies CLAHE slice-wise across the 3D MRI volume."""
        output = np.zeros_like(data_3d, dtype=np.float32)
        for z in range(data_3d.shape[2]):
            output[:, :, z] = cls.clahe_slice(data_3d[:, :, z], clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return output

    @classmethod
    def apply_he_volume(cls, data_3d: np.ndarray) -> np.ndarray:
        """Applies Global Histogram Equalization slice-wise across the 3D volume."""
        output = np.zeros_like(data_3d, dtype=np.float32)
        for z in range(data_3d.shape[2]):
            output[:, :, z] = cls.histogram_equalization_slice(data_3d[:, :, z])
        return output

    @classmethod
    def apply_ahe_volume(cls, data_3d: np.ndarray) -> np.ndarray:
        """Applies AHE slice-wise across the 3D volume."""
        output = np.zeros_like(data_3d, dtype=np.float32)
        for z in range(data_3d.shape[2]):
            output[:, :, z] = cls.ahe_slice(data_3d[:, :, z])
        return output
