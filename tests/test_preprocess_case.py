from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.preprocess_case import build_preprocess_transform


def test_preprocess_transform_is_binary_and_normalized(tmp_path: Path) -> None:
    shape = (16, 14, 12)
    affine = np.diag([0.5, 1.0, 2.0, 1.0])
    image = np.linspace(-1500, 2500, np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[3:12, 4:10, 2:9] = 7

    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)

    transform = build_preprocess_transform((1.0, 1.0, 1.0))
    result = transform({"image": str(image_path), "label": str(label_path)})

    output_image = result["image"]
    output_label = result["label"]

    assert output_image.ndim == 4
    assert output_label.shape == output_image.shape
    assert float(output_image.min()) >= 0.0
    assert float(output_image.max()) <= 1.0
    assert set(int(v) for v in np.unique(output_label)) <= {0, 1}
