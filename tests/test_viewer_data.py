from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.viewer_data import (
    build_surface_meshes,
    default_slice_indices,
    find_companion_label,
    load_case,
    orthogonal_slice,
    vertebra_name,
    vertebra_region,
)


def write_viewer_case(root: Path) -> tuple[Path, Path]:
    image_dir = root / "rawdata" / "sub-test001"
    label_dir = root / "derivatives" / "sub-test001"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    shape = (28, 24, 20)
    affine = np.diag([0.8, 1.2, 1.8, 1.0])
    image = np.full(shape, -1000, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    image[6:14, 5:13, 4:11] = 500
    image[15:23, 11:20, 10:17] = 650
    label[7:13, 6:12, 5:10] = 7
    label[16:22, 12:19, 11:16] = 20

    image_path = image_dir / "sub-test001_ct.nii.gz"
    label_path = label_dir / "sub-test001_seg-vert_msk.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    return image_path, label_path


def test_vertebra_names_and_regions() -> None:
    assert vertebra_name(1) == "C1"
    assert vertebra_name(19) == "T12"
    assert vertebra_name(24) == "L5"
    assert vertebra_name(28) == "T13"
    assert vertebra_region(7) == "Cervical"
    assert vertebra_region(8) == "Thoracic"
    assert vertebra_region(20) == "Lumbar"


def test_auto_match_load_and_navigate_case(tmp_path: Path) -> None:
    image_path, label_path = write_viewer_case(tmp_path)

    assert find_companion_label(image_path) == label_path.resolve()
    case = load_case(image_path)

    assert case.shape == (28, 24, 20)
    assert np.allclose(case.spacing_mm, (0.8, 1.2, 1.8))
    assert case.orientation == "RAS"
    assert case.labels == (7, 20)
    assert case.label_path == label_path.resolve()
    assert default_slice_indices(case) == (14, 12, 10)
    assert orthogonal_slice(case.image, 0, 14).shape == (20, 24)
    assert orthogonal_slice(case.image, 1, 12).shape == (20, 28)
    assert orthogonal_slice(case.image, 2, 10).shape == (24, 28)


def test_builds_one_surface_per_vertebra(tmp_path: Path) -> None:
    image_path, _label_path = write_viewer_case(tmp_path)
    case = load_case(image_path)

    meshes = build_surface_meshes(case)

    assert tuple(mesh.label for mesh in meshes) == (7, 20)
    assert all(mesh.vertices_mm.shape[1] == 3 for mesh in meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in meshes)
