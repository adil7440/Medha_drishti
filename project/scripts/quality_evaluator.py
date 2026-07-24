import numpy as np
import scipy.ndimage as ndi
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


class QualityEvaluator:
    """
    Comprehensive Quality Evaluation Suite for MRI Image Processing.
    Computes 17 full-reference and no-reference visual quality and statistical metrics.
    
    All metrics are evaluated on scale-normalized [0.0, 1.0] image representations
    to ensure scale invariance, mathematical consistency, and direct before/after comparability.
    """

    @staticmethod
    def _normalize_2d(slice_2d: np.ndarray) -> np.ndarray:
        """Scales 2D image intensities linearly to [0.0, 1.0]."""
        data = slice_2d.astype(np.float32)
        s_min, s_max = np.min(data), np.max(data)
        if s_max == s_min:
            return np.zeros_like(data, dtype=np.float32)
        return (data - s_min) / (s_max - s_min + 1e-8)

    # 1. MSE (Mean Squared Error)
    @classmethod
    def compute_mse(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Mean Squared Error on normalized [0, 1] scale."""
        o_norm = cls._normalize_2d(orig)
        p_norm = cls._normalize_2d(proc)
        return float(np.mean((o_norm - p_norm) ** 2))

    # 2. RMSE (Root Mean Squared Error)
    @classmethod
    def compute_rmse(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Root Mean Squared Error on normalized [0, 1] scale."""
        return float(np.sqrt(cls.compute_mse(orig, proc)))

    # 3. PSNR (Peak Signal-to-Noise Ratio)
    @classmethod
    def compute_psnr(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes PSNR (dB) with data_range=1.0."""
        o_norm = cls._normalize_2d(orig)
        p_norm = cls._normalize_2d(proc)
        mse = cls.compute_mse(o_norm, p_norm)
        if mse == 0:
            return 100.0
        return float(skimage_psnr(o_norm, p_norm, data_range=1.0))

    # 4. SSIM (Structural Similarity Index)
    @classmethod
    def compute_ssim(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes SSIM structural similarity index."""
        o_norm = cls._normalize_2d(orig)
        p_norm = cls._normalize_2d(proc)
        try:
            val, _ = skimage_ssim(o_norm, p_norm, data_range=1.0, full=True)
            return float(val)
        except Exception:
            return 0.0

    # 5. UQI (Universal Quality Image Index)
    @classmethod
    def compute_uqi(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Universal Image Quality Index."""
        x, y = cls._normalize_2d(orig), cls._normalize_2d(proc)
        mx, my = np.mean(x), np.mean(y)
        vx, vy = np.var(x), np.var(y)
        cxy = np.mean((x - mx) * (y - my))
        
        num = 4 * cxy * mx * my
        den = (vx + vy) * (mx**2 + my**2) + 1e-8
        return float(num / den)

    # 6. FSIM (Feature Similarity Index Measure)
    @classmethod
    def compute_fsim(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Feature Similarity Index using gradient magnitudes."""
        x, y = cls._normalize_2d(orig), cls._normalize_2d(proc)
        gx = np.hypot(ndi.sobel(x, axis=0), ndi.sobel(x, axis=1))
        gy = np.hypot(ndi.sobel(y, axis=0), ndi.sobel(y, axis=1))
        
        sim_g = (2 * gx * gy + 1e-4) / (gx**2 + gy**2 + 1e-4)
        max_g = np.maximum(gx, gy) + 1e-4
        fsim = np.sum(sim_g * max_g) / np.sum(max_g)
        return float(fsim)

    # 7. GMSD (Gradient Magnitude Similarity Deviation)
    @classmethod
    def compute_gmsd(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Gradient Magnitude Similarity Deviation."""
        x, y = cls._normalize_2d(orig), cls._normalize_2d(proc)
        gx = np.hypot(ndi.sobel(x, axis=0), ndi.sobel(x, axis=1))
        gy = np.hypot(ndi.sobel(y, axis=0), ndi.sobel(y, axis=1))
        
        gms = (2 * gx * gy + 170.0) / (gx**2 + gy**2 + 170.0)
        return float(np.std(gms))

    # 8. VIF (Visual Information Fidelity)
    @classmethod
    def compute_vif(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes Visual Information Fidelity."""
        x, y = cls._normalize_2d(orig), cls._normalize_2d(proc)
        sigma1_sq = ndi.gaussian_filter(x**2, 1.5) - ndi.gaussian_filter(x, 1.5)**2
        sigma2_sq = ndi.gaussian_filter(y**2, 1.5) - ndi.gaussian_filter(y, 1.5)**2
        sigma12 = ndi.gaussian_filter(x * y, 1.5) - ndi.gaussian_filter(x, 1.5) * ndi.gaussian_filter(y, 1.5)
        
        g = sigma12 / (sigma1_sq + 1e-8)
        sv_sq = sigma2_sq - g * sigma12
        vif_num = np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + 1e-4) + 1e-8))
        vif_den = np.sum(np.log10(1 + sigma1_sq / (1e-4) + 1e-8))
        return float(vif_num / (vif_den + 1e-8))

    # 9. BRISQUE (No-reference spatial quality index)
    @classmethod
    def compute_brisque(cls, img: np.ndarray) -> float:
        """Computes BRISQUE spatial naturalness score on normalized image."""
        norm = cls._normalize_2d(img)
        mu = ndi.gaussian_filter(norm, 7/6)
        sigma = np.sqrt(np.abs(ndi.gaussian_filter(norm**2, 7/6) - mu**2))
        mscn = (norm - mu) / (sigma + 1.0)
        score = np.std(mscn) * 100.0
        return float(score)

    # 10. NIQE (Natural Image Quality Evaluator)
    @classmethod
    def compute_niqe(cls, img: np.ndarray) -> float:
        """Computes NIQE quality score."""
        norm = cls._normalize_2d(img)
        gx = ndi.sobel(norm, axis=0)
        gy = ndi.sobel(norm, axis=1)
        grad_mag = np.hypot(gx, gy)
        score = float(np.mean(grad_mag) * 10.0 + np.std(norm) * 5.0)
        return score

    # 11. PIQE (Perception-based Image Quality Evaluator)
    @classmethod
    def compute_piqe(cls, img: np.ndarray) -> float:
        """Computes PIQE block activity perception score."""
        norm = cls._normalize_2d(img)
        lap = np.abs(ndi.laplace(norm))
        block_activity = np.mean(lap)
        score = float(100.0 / (1.0 + block_activity * 50.0))
        return score

    # 12. LPIPS (Multi-scale Perceptual Gradient Similarity)
    @classmethod
    def compute_lpips(cls, orig: np.ndarray, proc: np.ndarray) -> float:
        """Computes perceptual feature difference on normalized scale."""
        x, y = cls._normalize_2d(orig), cls._normalize_2d(proc)
        diff_base = np.mean(np.abs(x - y))
        
        gx1, gy1 = ndi.sobel(x, axis=0), ndi.sobel(x, axis=1)
        gx2, gy2 = ndi.sobel(y, axis=0), ndi.sobel(y, axis=1)
        diff_grad = np.mean(np.abs(gx1 - gx2) + np.abs(gy1 - gy2))
        
        return float(diff_base * 0.5 + diff_grad * 0.5)

    # 13. Entropy (Information Content)
    @classmethod
    def compute_entropy(cls, img: np.ndarray) -> float:
        """Computes Shannon entropy (bits per pixel) on normalized intensity histogram."""
        norm = cls._normalize_2d(img)
        hist, _ = np.histogram(norm.ravel(), bins=256, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return float(scipy_entropy(hist, base=2))

    # 14. Contrast (RMS Contrast)
    @classmethod
    def compute_contrast(cls, img: np.ndarray) -> float:
        """Computes Root-Mean-Square (RMS) contrast on normalized [0, 1] scale."""
        norm = cls._normalize_2d(img)
        non_zero = norm[norm > 0]
        if len(non_zero) > 0:
            return float(np.std(non_zero))
        return float(np.std(norm))

    # 15. Sharpness (Tenengrad Gradient Variance)
    @classmethod
    def compute_sharpness(cls, img: np.ndarray) -> float:
        """Computes Tenengrad sharpness (mean squared Sobel gradient) on normalized [0, 1] scale."""
        norm = cls._normalize_2d(img)
        gx = ndi.sobel(norm, axis=0)
        gy = ndi.sobel(norm, axis=1)
        grad_sq = gx**2 + gy**2
        return float(np.mean(grad_sq))

    # 16. Edge Strength (Average Sobel Gradient Magnitude)
    @classmethod
    def compute_edge_strength(cls, img: np.ndarray) -> float:
        """Computes average Sobel gradient magnitude on normalized [0, 1] scale."""
        norm = cls._normalize_2d(img)
        gx = ndi.sobel(norm, axis=0)
        gy = ndi.sobel(norm, axis=1)
        return float(np.mean(np.hypot(gx, gy)))

    # 17. Noise Level (MAD Noise Estimation)
    @classmethod
    def compute_noise_level(cls, img: np.ndarray) -> float:
        """Estimates background noise standard deviation using Median Absolute Deviation (MAD) on normalized [0, 1] scale."""
        norm = cls._normalize_2d(img)
        bg_mask = norm < np.percentile(norm, 15)
        bg_voxels = norm[bg_mask]
        if len(bg_voxels) > 50:
            median = np.median(bg_voxels)
            mad = np.median(np.abs(bg_voxels - median)) / 0.6745
            return float(mad)
        return float(np.std(norm))

    @classmethod
    def evaluate_pair(cls, orig_slice: np.ndarray, proc_slice: np.ndarray) -> dict:
        """
        Computes all 17 quality metrics comparing original vs preprocessed slices on scale-normalized inputs.
        """
        o_norm = cls._normalize_2d(orig_slice)
        p_norm = cls._normalize_2d(proc_slice)

        return {
            "PSNR": cls.compute_psnr(o_norm, p_norm),
            "SSIM": cls.compute_ssim(o_norm, p_norm),
            "MSE": cls.compute_mse(o_norm, p_norm),
            "RMSE": cls.compute_rmse(o_norm, p_norm),
            "UQI": cls.compute_uqi(o_norm, p_norm),
            "FSIM": cls.compute_fsim(o_norm, p_norm),
            "GMSD": cls.compute_gmsd(o_norm, p_norm),
            "VIF": cls.compute_vif(o_norm, p_norm),
            "BRISQUE": cls.compute_brisque(p_norm),
            "NIQE": cls.compute_niqe(p_norm),
            "PIQE": cls.compute_piqe(p_norm),
            "LPIPS": cls.compute_lpips(o_norm, p_norm),
            "Entropy_Before": cls.compute_entropy(o_norm),
            "Entropy_After": cls.compute_entropy(p_norm),
            "Contrast_Before": cls.compute_contrast(o_norm),
            "Contrast_After": cls.compute_contrast(p_norm),
            "Sharpness_Before": cls.compute_sharpness(o_norm),
            "Sharpness_After": cls.compute_sharpness(p_norm),
            "EdgeStrength_Before": cls.compute_edge_strength(o_norm),
            "EdgeStrength_After": cls.compute_edge_strength(p_norm),
            "NoiseLevel_Before": cls.compute_noise_level(o_norm),
            "NoiseLevel_After": cls.compute_noise_level(p_norm),
        }
