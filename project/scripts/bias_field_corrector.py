import numpy as np
import SimpleITK as sitk


class N4BiasFieldCorrector:
    """
    Applies N4 Bias Field Correction (SimpleITK) to correct low-frequency 
    intensity non-uniformities caused by RF coil inhomogeneity in MRI.
    """

    @staticmethod
    def correct_sitk(
        sitk_image: sitk.Image,
        mask_image: sitk.Image = None,
        shrink_factor: int = 2,
        maximum_number_of_iterations: list = [50, 50, 30]
    ) -> sitk.Image:
        """
        Executes N4 Bias Field Correction on a SimpleITK Image.
        """
        # Ensure float pixel type required by N4 filter
        input_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

        # Shrink image for computational speed if requested
        if shrink_factor > 1:
            input_subsampled = sitk.Shrink(input_image, [shrink_factor] * input_image.GetDimension())
            if mask_image is not None:
                mask_subsampled = sitk.Shrink(mask_image, [shrink_factor] * mask_image.GetDimension())
            else:
                mask_subsampled = sitk.OtsuThreshold(input_subsampled, 0, 1, 200)
        else:
            input_subsampled = input_image
            mask_subsampled = mask_image if mask_image is not None else sitk.OtsuThreshold(input_image, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(maximum_number_of_iterations)
        
        # Fit bias field on subsampled image
        _ = corrector.Execute(input_subsampled, mask_subsampled)

        # Evaluate reconstructed bias field at full resolution
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
        corrected_full_res = input_image / sitk.Exp(log_bias_field)

        return corrected_full_res

    @classmethod
    def correct_numpy(cls, data_3d: np.ndarray, shrink_factor: int = 2) -> np.ndarray:
        """
        Applies N4 Bias Field Correction on a 3D numpy array.
        """
        sitk_img = sitk.GetImageFromArray(data_3d.astype(np.float32))
        corrected_sitk = cls.correct_sitk(sitk_img, shrink_factor=shrink_factor)
        return sitk.GetArrayFromImage(corrected_sitk)
