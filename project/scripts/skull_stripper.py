import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing, binary_opening, remove_small_objects


class SkullStripper:
    """
    Performs non-deep-learning Skull Stripping for Brain MRI datasets using:
    - Otsu adaptive intensity thresholding
    - 3D Morphological connected component extraction
    - Hole filling & binary closing/opening
    (Skipped automatically for Spine datasets).
    """

    @staticmethod
    def extract_brain_mask(data_3d: np.ndarray) -> np.ndarray:
        """
        Generates a 3D binary brain tissue mask from a 3D MRI volume.
        """
        data = data_3d.astype(np.float32)
        non_zero = data[data > 0]
        if len(non_zero) == 0:
            return np.zeros_like(data_3d, dtype=bool)

        # 1. Otsu thresholding on non-background voxels
        thresh = threshold_otsu(non_zero)
        binary_mask = data > (thresh * 0.3)  # Low threshold to capture brain tissue

        # 2. Extract largest 3D connected component
        labeled_mask, num_features = ndi.label(binary_mask)
        if num_features == 0:
            return binary_mask

        sizes = ndi.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        largest_component_label = np.argmax(sizes) + 1
        brain_mask = (labeled_mask == largest_component_label)

        # 3. Morphological 3D hole filling & smoothing
        brain_mask = ndi.binary_fill_holes(brain_mask)
        brain_mask = binary_closing(brain_mask, footprint=ball(3))
        brain_mask = binary_opening(brain_mask, footprint=ball(2))

        return brain_mask

    @classmethod
    def apply_skull_stripping(cls, data_3d: np.ndarray, is_spine: bool = False) -> tuple:
        """
        Applies skull stripping if dataset is Brain; skips if dataset is Spine.
        Returns: (stripped_volume, brain_mask)
        """
        if is_spine:
            # Skip for Spine dataset as specified
            mask = np.ones_like(data_3d, dtype=bool)
            return data_3d, mask

        brain_mask = cls.extract_brain_mask(data_3d)
        stripped = data_3d.copy()
        stripped[~brain_mask] = 0
        return stripped, brain_mask
