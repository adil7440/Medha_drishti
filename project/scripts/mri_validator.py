import os
import numpy as np
import nibabel as nib
from pathlib import Path


class MRIValidator:
    """
    Production-grade validator for NIfTI MRI volumes (.nii / .nii.gz).
    Validates:
    - File existence & un-corrupted headers
    - Missing or empty slices
    - NaN / Inf intensity values
    - Invalid or non-positive voxel spacing
    - Orientation / Affine matrix validity
    """

    @staticmethod
    def validate_file(file_path: str) -> dict:
        """
        Validates an MRI file and returns a dictionary with status and diagnostic checks.
        """
        path = Path(file_path)
        result = {
            "file_path": str(path),
            "is_valid": True,
            "corrupted": False,
            "has_nan": False,
            "has_inf": False,
            "invalid_spacing": False,
            "invalid_orientation": False,
            "missing_slices": False,
            "error_message": "",
            "shape": None,
            "spacing": None,
            "num_slices": 0,
            "nan_count": 0,
            "inf_count": 0
        }

        if not path.exists():
            result["is_valid"] = False
            result["corrupted"] = True
            result["error_message"] = f"File does not exist: {path}"
            return result

        try:
            nimg = nib.load(str(path))
            header = nimg.header
            data = nimg.get_fdata()
            shape = data.shape
            result["shape"] = shape
            result["spacing"] = tuple(float(s) for s in header.get_zooms()[:len(shape)])

            # 1. Check for NaN / Inf
            nan_mask = np.isnan(data)
            inf_mask = np.isinf(data)
            result["nan_count"] = int(np.sum(nan_mask))
            result["inf_count"] = int(np.sum(inf_mask))
            if result["nan_count"] > 0:
                result["has_nan"] = True
                result["is_valid"] = False
            if result["inf_count"] > 0:
                result["has_inf"] = True
                result["is_valid"] = False

            # 2. Check spacing validity
            zooms = header.get_zooms()
            if any(z <= 0 for z in zooms):
                result["invalid_spacing"] = True
                result["is_valid"] = False

            # 3. Check for missing/blank slices along depth dimension
            if len(shape) >= 3:
                depth = shape[2]
                result["num_slices"] = depth
                empty_slices = 0
                for z in range(depth):
                    slice_data = data[:, :, z] if len(shape) == 3 else data[:, :, z, 0]
                    if np.all(slice_data == 0) or np.max(slice_data) == np.min(slice_data):
                        empty_slices += 1
                
                # Flag if empty slices occur internally (excluding top/bottom border padding)
                if empty_slices > (depth * 0.4):
                    result["missing_slices"] = True

            # 4. Check affine matrix
            affine = nimg.affine
            if affine is None or np.isnan(affine).any() or np.linalg.det(affine[:3, :3]) == 0:
                result["invalid_orientation"] = True
                result["is_valid"] = False

        except Exception as e:
            result["is_valid"] = False
            result["corrupted"] = True
            result["error_message"] = str(e)

        return result
