import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.build_inventory import build_inventory, find_pairs


def write_case(root: Path, subject: str, label_value: int) -> None:
    shape = (12, 10, 8)
    affine = np.diag([0.5, 0.75, 1.25, 1.0])
    image = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[2:8, 3:7, 1:6] = label_value

    raw_dir = root / "rawdata" / subject
    derivatives_dir = root / "derivatives" / subject
    raw_dir.mkdir(parents=True)
    derivatives_dir.mkdir(parents=True)

    nib.save(nib.Nifti1Image(image, affine), raw_dir / f"{subject}_ct.nii.gz")
    nib.save(
        nib.Nifti1Image(label, affine),
        derivatives_dir / f"{subject}_seg-vert_msk.nii.gz",
    )


def test_build_inventory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_case(dataset_root, "sub-test001", 3)
    write_case(dataset_root, "sub-test002", 20)
    output = tmp_path / "inventory.csv"

    assert len(find_pairs(dataset_root)) == 2
    records = build_inventory(dataset_root, output)

    assert len(records) == 2
    assert records[0].spacing_z_mm == 1.25
    assert records[0].label_count == 1
    assert records[1].labels == "0;20"

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["aligned"] == "True"
