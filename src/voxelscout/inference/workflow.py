"""CT-only orchestration shared by the GUI worker and unit tests."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.desktop_data import (
    ProgressCallback,
    SegmentationVolume,
    SegmentedCase,
    build_segmented_case,
    find_companion_mask,
    load_ct_volume,
    load_segmentation_volume,
)
from voxelscout.inference.backend import SegmentationBackend
from voxelscout.inference.nnunet_backend import NnUNetBackend


def default_cache_directory() -> Path:
    configured = os.environ.get("VOXELSCOUT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "VoxelScout" / "predictions"


def _prediction_cache_path(
    ct_path: Path, backend: SegmentationBackend, cache_dir: Path
) -> Path:
    path = Path(ct_path).resolve()
    stat = path.stat()
    identity = "|".join(
        (str(path), str(stat.st_mtime_ns), str(stat.st_size), backend.name, backend.cache_key)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.nii.gz"


def _validate_prediction_geometry(
    ct_shape: tuple[int, ...], ct_affine: np.ndarray, segmentation: SegmentationVolume
) -> None:
    label_shape = tuple(int(value) for value in segmentation.labels.shape)
    if tuple(ct_shape) != label_shape:
        raise ValueError(
            f"CT shape {tuple(ct_shape)} does not match mask shape {label_shape}"
        )
    if not np.allclose(ct_affine, segmentation.affine, atol=1e-3, rtol=1e-5):
        raise ValueError("CT and segmentation do not use the same spatial coordinates")


def _write_cached_prediction(
    path: Path, segmentation: SegmentationVolume
) -> SegmentationVolume:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.asarray(segmentation.labels, dtype=np.uint8), segmentation.affine
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".nii.gz", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return SegmentationVolume(
        labels=np.asarray(segmentation.labels, dtype=np.uint8),
        affine=np.asarray(segmentation.affine, dtype=float),
        source=segmentation.source,
        source_path=path,
    )


def load_case_for_ct(
    ct_path: Path,
    mask_path: Path | None = None,
    *,
    backend: SegmentationBackend | None = None,
    cache_dir: Path | None = None,
    sample_step: int = 2,
    target_faces_per_vertebra: int = 12_000,
    progress: ProgressCallback | None = None,
) -> SegmentedCase:
    """Use a trusted mask when available, otherwise predict and cache one."""
    report = progress or (lambda _value, _message: None)
    ct_path = Path(ct_path).resolve()
    ct = load_ct_volume(ct_path, progress=report)
    selected_mask = Path(mask_path).resolve() if mask_path is not None else None
    if selected_mask is None:
        selected_mask = find_companion_mask(ct_path)

    if selected_mask is not None:
        report(12, "Using companion mask")
        segmentation = load_segmentation_volume(selected_mask)
    else:
        predictor = backend or NnUNetBackend.from_environment()
        prediction_path = _prediction_cache_path(
            ct_path, predictor, cache_dir or default_cache_directory()
        )
        if prediction_path.is_file():
            report(16, "Using cached prediction")
            segmentation = load_segmentation_volume(
                prediction_path,
                source=f"prediction-cache:{predictor.name}:{predictor.cache_key}",
            )
        else:
            report(16, "Running pretrained nnU-Net model")
            segmentation = predictor.predict(ct_path, progress=report)
            report(72, "Real model inference · validating prediction")
            _validate_prediction_geometry(ct.data.shape, ct.affine, segmentation)
            segmentation = SegmentationVolume(
                labels=segmentation.labels,
                affine=segmentation.affine,
                source=f"prediction-cache:{predictor.name}:{predictor.cache_key}",
            )
            segmentation = _write_cached_prediction(prediction_path, segmentation)

    def mesh_progress(value: int, message: str) -> None:
        report(75 + int(max(0, min(100, value)) * 0.25), message)

    return build_segmented_case(
        ct,
        segmentation,
        sample_step=sample_step,
        target_faces_per_vertebra=target_faces_per_vertebra,
        progress=mesh_progress,
    )
