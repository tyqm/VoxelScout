"""CT-first orchestration shared by the GUI worker and unit tests."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.desktop_data import (
    ProgressCallback,
    SegmentationVolume,
    SegmentedCase,
    build_segmented_case,
    load_ct_volume,
    load_segmentation_volume,
)
from voxelscout.inference.backend import SegmentationBackend
from voxelscout.inference.metrics import evaluate_segmentation
from voxelscout.inference.nnunet_backend import NnUNetBackend


def default_cache_directory() -> Path:
    configured = os.environ.get("VOXELSCOUT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "VoxelScout" / "predictions"


def _prediction_cache_path(ct_path: Path, backend: SegmentationBackend, cache_dir: Path) -> Path:
    path = Path(ct_path).resolve()
    stat = path.stat()
    identity = "|".join(
        (str(path), str(stat.st_mtime_ns), str(stat.st_size), backend.name, backend.cache_key)
    )
    return Path(cache_dir) / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.nii.gz"


def _validate_prediction_geometry(
    ct_shape: tuple[int, ...], ct_affine: np.ndarray, segmentation: SegmentationVolume
) -> None:
    label_shape = tuple(int(value) for value in segmentation.labels.shape)
    if tuple(ct_shape) != label_shape:
        raise ValueError(f"CT shape {tuple(ct_shape)} does not match mask shape {label_shape}")
    if not np.allclose(ct_affine, segmentation.affine, atol=1e-3, rtol=1e-5):
        raise ValueError("CT and segmentation do not use the same spatial coordinates")


def _write_cached_prediction(path: Path, segmentation: SegmentationVolume) -> SegmentationVolume:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(segmentation.labels, dtype=np.uint8), segmentation.affine)
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
    ground_truth_path: Path | None = None,
    *,
    backend: SegmentationBackend | None = None,
    cache_dir: Path | None = None,
    sample_step: int = 2,
    target_faces_per_vertebra: int = 12_000,
    progress: ProgressCallback | None = None,
) -> SegmentedCase:
    """Always obtain a prediction; ground truth is used only for evaluation."""
    report = progress or (lambda _value, _message: None)
    ct_path = Path(ct_path).resolve()
    report(2, "Loading CT")
    ct = load_ct_volume(ct_path, progress=report)
    predictor = backend or NnUNetBackend.from_environment()
    prediction_path = _prediction_cache_path(
        ct_path, predictor, cache_dir or default_cache_directory()
    )
    inference_time: float | None = None
    peak_memory: float | None = None
    if prediction_path.is_file():
        report(16, "Cached prediction")
        segmentation = load_segmentation_volume(
            prediction_path,
            source=f"prediction-cache:{predictor.name}:{predictor.cache_key}",
        )
        status = "Cached prediction"
    else:
        report(16, "Running segmentation")
        started = time.perf_counter()
        segmentation = predictor.predict(ct_path, progress=report)
        inference_time = time.perf_counter() - started
        peak_memory = getattr(predictor, "last_peak_memory_mib", None)
        report(72, "Validating prediction")
        _validate_prediction_geometry(ct.data.shape, ct.affine, segmentation)
        segmentation = SegmentationVolume(
            labels=segmentation.labels,
            affine=segmentation.affine,
            source=f"prediction-cache:{predictor.name}:{predictor.cache_key}",
        )
        segmentation = _write_cached_prediction(prediction_path, segmentation)
        status = "Complete"

    metrics = None
    if ground_truth_path is not None:
        reference = load_segmentation_volume(Path(ground_truth_path), source="ground-truth")
        _validate_prediction_geometry(ct.data.shape, ct.affine, reference)
        metrics = evaluate_segmentation(segmentation.labels, reference.labels, ct.affine)

    def mesh_progress(value: int, _message: str) -> None:
        report(75 + int(max(0, min(100, value)) * 0.25), "Generating 3D model")

    case = build_segmented_case(
        ct,
        segmentation,
        sample_step=sample_step,
        target_faces_per_vertebra=target_faces_per_vertebra,
        progress=mesh_progress,
        model_name=predictor.name,
        segmentation_status=status,
        inference_time_seconds=inference_time,
        peak_memory_mib=peak_memory,
        dice=metrics.dice if metrics else None,
        iou=metrics.iou if metrics else None,
        hd95_mm=metrics.hd95_mm if metrics else None,
    )
    report(100, "Complete")
    return case
