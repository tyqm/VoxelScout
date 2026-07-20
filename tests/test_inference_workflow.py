from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from voxelscout.desktop_data import (
    CTVolume,
    SegmentationVolume,
    build_segmented_case,
)
from voxelscout.inference.backend import InferenceUnavailableError
from voxelscout.inference.labels import training_to_verse_labels
from voxelscout.inference.nnunet_backend import NnUNetConfig
from voxelscout.inference.workflow import load_case_for_ct


def write_ct(path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    shape = (24, 22, 20)
    affine = np.diag([0.8, 1.1, 1.6, 1.0])
    data = np.full(shape, -1000, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path, data, affine


def labelled_mask(shape: tuple[int, ...], label: int = 20) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:15, 6:16, 4:14] = label
    return mask


def volumes(tmp_path: Path) -> tuple[CTVolume, SegmentationVolume]:
    path, data, affine = write_ct(tmp_path / "scan_ct.nii.gz")
    ct = CTVolume(
        data=data,
        affine=affine,
        spacing=(0.8, 1.1, 1.6),
        orientation="RAS",
        source_path=path,
    )
    segmentation = SegmentationVolume(
        labels=labelled_mask(data.shape),
        affine=affine,
        source="synthetic",
    )
    return ct, segmentation


def test_build_segmented_case_accepts_in_memory_volumes(tmp_path: Path) -> None:
    ct, segmentation = volumes(tmp_path)

    case = build_segmented_case(ct, segmentation, sample_step=1)

    assert case.labels == (20,)
    assert case.shape == ct.data.shape


@pytest.mark.parametrize("problem", ["shape", "affine"])
def test_build_segmented_case_rejects_geometry_mismatch(
    tmp_path: Path, problem: str
) -> None:
    ct, segmentation = volumes(tmp_path)
    if problem == "shape":
        segmentation = SegmentationVolume(
            labels=np.zeros((8, 8, 8), dtype=np.uint8),
            affine=segmentation.affine,
            source="synthetic",
        )
    else:
        shifted = segmentation.affine.copy()
        shifted[0, 3] = 3.0
        segmentation = SegmentationVolume(
            labels=segmentation.labels,
            affine=shifted,
            source="synthetic",
        )

    with pytest.raises(ValueError, match="shape|spatial coordinates"):
        build_segmented_case(ct, segmentation)


def test_prediction_label_26_maps_back_to_verse_28() -> None:
    prediction = np.asarray([0, 1, 25, 26], dtype=np.uint8)

    converted = training_to_verse_labels(prediction)

    assert converted.tolist() == [0, 1, 25, 28]


class FakeBackend:
    name = "fake"
    cache_key = "test-model-v1"

    def __init__(self, affine: np.ndarray, shape: tuple[int, ...]) -> None:
        self.affine = affine
        self.shape = shape
        self.calls = 0

    def predict(self, ct_path: Path, *, progress=None) -> SegmentationVolume:
        self.calls += 1
        if progress is not None:
            progress(25, "Running vertebra segmentation")
        return SegmentationVolume(
            labels=labelled_mask(self.shape, 28),
            affine=self.affine,
            source="fake:test-model-v1",
        )


def test_companion_mask_bypasses_inference_backend(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    ct_dir = root / "rawdata" / "sub-test"
    mask_dir = root / "derivatives" / "sub-test"
    ct_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    ct_path, data, affine = write_ct(ct_dir / "sub-test_ct.nii.gz")
    nib.save(
        nib.Nifti1Image(labelled_mask(data.shape), affine),
        mask_dir / "sub-test_seg-vert_msk.nii.gz",
    )
    backend = FakeBackend(affine, data.shape)
    updates: list[str] = []

    case = load_case_for_ct(
        ct_path,
        backend=backend,
        cache_dir=tmp_path / "cache",
        progress=lambda _value, message: updates.append(message),
    )

    assert backend.calls == 0
    assert case.labels == (20,)
    assert "Using companion mask" in updates


def test_missing_companion_uses_backend_and_existing_mesh_pipeline(
    tmp_path: Path,
) -> None:
    ct_path, data, affine = write_ct(tmp_path / "standalone_ct.nii.gz")
    backend = FakeBackend(affine, data.shape)
    updates: list[str] = []
    cached_updates: list[str] = []

    first = load_case_for_ct(
        ct_path,
        backend=backend,
        cache_dir=tmp_path / "cache",
        sample_step=1,
        progress=lambda _value, message: updates.append(message),
    )
    second = load_case_for_ct(
        ct_path,
        backend=backend,
        cache_dir=tmp_path / "cache",
        sample_step=1,
        progress=lambda _value, message: cached_updates.append(message),
    )

    assert backend.calls == 1
    assert first.labels == (28,)
    assert second.labels == (28,)
    assert "Running pretrained nnU-Net model" in updates
    assert "Running vertebra segmentation" in updates
    assert "Using cached prediction" in cached_updates
    assert list((tmp_path / "cache").glob("*.nii.gz"))


def test_backend_unavailable_has_clear_error(tmp_path: Path) -> None:
    config = NnUNetConfig(
        model_dir=tmp_path / "missing-model",
        command="voxelscout-command-that-does-not-exist",
    )

    with pytest.raises(InferenceUnavailableError, match="command was not found"):
        config.validate()
