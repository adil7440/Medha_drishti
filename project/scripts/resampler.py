import numpy as np
import SimpleITK as sitk
import nibabel as nib


class MRIResampler:
    """
    Resamples NIfTI MRI volumes to isotropic voxel dimensions (e.g., 1.0mm x 1.0mm x 1.0mm)
    using SimpleITK spline/linear interpolation.
    """

    @staticmethod
    def resample_sitk(sitk_image: sitk.Image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear) -> sitk.Image:
        """
        Resamples a SimpleITK image to the specified target isotropic voxel spacing.
        """
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()

        # Compute new dimensions based on ratio of original spacing to new spacing
        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(len(original_size))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetDefaultPixelValue(0)

        return resample.Execute(sitk_image)

    @classmethod
    def resample_numpy(cls, data_3d: np.ndarray, current_spacing=(1.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Resamples a 3D numpy array using SimpleITK resampling wrapper.
        """
        sitk_img = sitk.GetImageFromArray(data_3d.astype(np.float32))
        sitk_img.SetSpacing(current_spacing)
        resampled_sitk = cls.resample_sitk(sitk_img, new_spacing=target_spacing)
        return sitk.GetArrayFromImage(resampled_sitk)
