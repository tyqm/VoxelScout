from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from voxelscout.anatomy import vertebra_info
from voxelscout.desktop_data import find_companion_mask, load_segmented_case


def write_case(root: Path) -> tuple[Path, Path]:
    image_dir = root / "rawdata" / "sub-test001"
    mask_dir = root / "derivatives" / "sub-test001"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    shape = (32, 28, 24)
    affine = np.diag([0.8, 1.1, 1.7, 1.0])
    image = np.full(shape, -1000, dtype=np.int16)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:13, 6:15, 4:12] = 7
    mask[17:27, 12:23, 11:21] = 20
    image[mask > 0] = 500

    image_path = image_dir / "sub-test001_ct.nii.gz"
    mask_path = mask_dir / "sub-test001_seg-vert_msk.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return image_path, mask_path


def test_plain_language_hover_text() -> None:
    info = vertebra_info(21)
    assert info.code == "L2"
    assert info.anatomical_name == "Second lumbar vertebra"
    assert info.region_plain == "Lower-back region"
    assert info.tooltip.splitlines() == [
        "L2",
        "Second lumbar vertebra",
        "Lower-back region",
    ]


def test_builds_and_caches_one_mesh_per_vertebra(tmp_path: Path) -> None:
    image_path, mask_path = write_case(tmp_path)
    updates = []

    case = load_segmented_case(
        image_path,
        mask_path,
        sample_step=1,
        target_faces_per_vertebra=2_000,
        progress=lambda value, message: updates.append((value, message)),
    )
    cached = load_segmented_case(
        image_path,
        mask_path,
        sample_step=1,
        target_faces_per_vertebra=2_000,
    )

    assert cached is case
    assert case.labels == (7, 20)
    assert case.shape == (32, 28, 24)
    assert np.allclose(case.spacing, (0.8, 1.1, 1.7))
    assert case.orientation == "RAS"
    assert case.mesh_memory_mib < 1
    assert all(mesh.vertices.shape[1] == 3 for mesh in case.meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in case.meshes)
    assert updates[-1] == (100, "3D spine ready")


def test_finds_verse_companion_mask(tmp_path: Path) -> None:
    image_path, mask_path = write_case(tmp_path)
    assert find_companion_mask(image_path) == mask_path


def test_rejects_misaligned_segmentation(tmp_path: Path) -> None:
    image_path, mask_path = write_case(tmp_path)
    mask = nib.load(str(mask_path))
    shifted = mask.affine.copy()
    shifted[0, 3] += 2
    nib.save(nib.Nifti1Image(np.asanyarray(mask.dataobj), shifted), mask_path)

    with pytest.raises(ValueError, match="spatial coordinates"):
        load_segmented_case(image_path, mask_path)


def test_retries_marching_cubes_at_full_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, mask_path = write_case(tmp_path)
    from skimage import measure

    original = measure.marching_cubes
    steps: list[int] = []

    def flaky_marching_cubes(*args, **kwargs):
        step_size = int(kwargs["step_size"])
        steps.append(step_size)
        if step_size > 1:
            raise ValueError("coarse sampling missed the surface")
        return original(*args, **kwargs)

    monkeypatch.setattr(measure, "marching_cubes", flaky_marching_cubes)

    case = load_segmented_case(image_path, mask_path, sample_step=3)

    assert case.labels == (7, 20)
    assert steps == [3, 1, 3, 1]


def test_skips_label_when_surface_generation_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, mask_path = write_case(tmp_path)
    from skimage import measure

    original = measure.marching_cubes

    def fail_for_first_label(volume, *args, **kwargs):
        if int(np.count_nonzero(volume)) == 8 * 9 * 8:
            raise ValueError("invalid predicted component")
        return original(volume, *args, **kwargs)

    monkeypatch.setattr(measure, "marching_cubes", fail_for_first_label)

    case = load_segmented_case(image_path, mask_path, sample_step=2)

    assert case.labels == (20,)


def test_ignores_labels_smaller_than_fifty_voxels(tmp_path: Path) -> None:
    image_path, mask_path = write_case(tmp_path)
    mask_image = nib.load(str(mask_path))
    mask = np.asanyarray(mask_image.dataobj).copy()
    mask[0:3, 0:4, 0:4] = 12  # 48 voxels
    nib.save(nib.Nifti1Image(mask, mask_image.affine), mask_path)

    case = load_segmented_case(image_path, mask_path, sample_step=1)

    assert case.labels == (7, 20)
