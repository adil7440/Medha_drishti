import numpy as np
import scipy.ndimage as ndi


class MRIAugmentor:
    """
    Data Augmentation pipeline for 3D/2D MRI volumes:
    - Random Rotation
    - Random Flip
    - Gamma Correction
    - Gaussian Noise Injection
    - Intensity Scaling
    - Elastic Deformation
    """

    @staticmethod
    def rotate(data_3d: np.ndarray, angle_deg: float = 10.0, axes=(0, 1)) -> np.ndarray:
        """Rotates the MRI volume by angle_deg degrees in the specified plane."""
        return ndi.rotate(data_3d.astype(np.float32), angle=angle_deg, axes=axes, reshape=False, mode='constant', cval=0)

    @staticmethod
    def flip(data_3d: np.ndarray, axis: int = 1) -> np.ndarray:
        """Flips the MRI volume along the given axis (0=coronal, 1=sagittal, 2=axial)."""
        return np.flip(data_3d, axis=axis)

    @staticmethod
    def gamma_correction(data_3d: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Applies non-linear gamma intensity transformation."""
        data = data_3d.astype(np.float32)
        dmin, dmax = np.min(data), np.max(data)
        if dmax == dmin:
            return data
        
        norm = (data - dmin) / (dmax - dmin + 1e-8)
        corrected = np.power(norm, gamma)
        return corrected * (dmax - dmin) + dmin

    @staticmethod
    def add_gaussian_noise(data_3d: np.ndarray, std: float = 0.02) -> np.ndarray:
        """Injects additive zero-mean Gaussian noise into the MRI volume."""
        data = data_3d.astype(np.float32)
        drange = np.max(data) - np.min(data)
        noise = np.random.normal(0, std * drange, size=data.shape)
        return np.clip(data + noise, np.min(data), np.max(data))

    @staticmethod
    def intensity_scale(data_3d: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """Scales intensity values by a linear multiplier factor."""
        return data_3d.astype(np.float32) * factor

    @staticmethod
    def elastic_transform(data_3d: np.ndarray, alpha: float = 30.0, sigma: float = 4.0) -> np.ndarray:
        """
        Applies 3D/2D Elastic Deformation using Gaussian displacement fields.
        """
        shape = data_3d.shape
        dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        if len(shape) == 3:
            dz = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices = (grid[0] + dx, grid[1] + dy, grid[2] + dz)
        else:
            grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = (grid[0] + dx, grid[1] + dy)

        return ndi.map_coordinates(data_3d.astype(np.float32), indices, order=1, mode='reflect')
