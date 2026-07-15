from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.inspect_case import (
    describe,
    foreground_centre,
    inspect_case,
    validate_pair,
)


def test_inspect_synthetic_case(tmp_path: Path) -> None:
    shape = (24, 20, 16)
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    ct = np.full(shape, -1000, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    ct[7:17, 6:15, 4:13] = 450
    mask[8:16, 7:14, 5:12] = 20

    image_path = tmp_path / "ct.nii.gz"
    label_path = tmp_path / "mask.nii.gz"
    output_path = tmp_path / "overlay.png"
    nib.save(nib.Nifti1Image(ct, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), label_path)

    inspect_case(image_path, label_path, output_path)

    assert output_path.is_file()
    assert foreground_centre(mask) == (11, 10, 8)


def test_volume_description_and_alignment() -> None:
    affine = np.eye(4)
    ct = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine)
    label_data = np.zeros((4, 5, 6), dtype=np.uint8)
    label_data[1, 2, 3] = 7
    label = nib.Nifti1Image(label_data, affine)

    validate_pair(ct, label)
    info = describe(label, is_label=True)

    assert info.shape == (4, 5, 6)
    assert info.labels == (0, 7)
