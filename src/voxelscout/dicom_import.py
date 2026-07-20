"""DICOM CT series discovery and privacy-preserving NIfTI conversion."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import nibabel as nib
import SimpleITK as sitk


ProgressCallback = Callable[[int, str], None]
_LOCALIZER_WORDS = ("localizer", "locator", "scout", "topogram", "survey")


@dataclass(frozen=True)
class DicomSeries:
    directory: Path
    series_id: str
    files: tuple[Path, ...]
    description: str
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    is_localizer: bool

    @property
    def display_name(self) -> str:
        description = self.description or "CT series"
        size = " × ".join(str(value) for value in self.shape)
        spacing = " × ".join(f"{value:g}" for value in self.spacing)
        return f"{description} — {size}, {spacing} mm"


@dataclass(frozen=True)
class ConvertedDicom:
    nifti_path: Path
    series: DicomSeries
    orientation: str
    cached: bool


def default_dicom_cache_directory() -> Path:
    configured = os.environ.get("VOXELSCOUT_DICOM_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "VoxelScout" / "dicom"


def discover_ct_series(folder: Path) -> list[DicomSeries]:
    """Discover CT volumes through GDCM and rank the likely diagnostic series."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"DICOM folder does not exist: {root}")

    candidates: list[DicomSeries] = []
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in directories:
        if not any(path.is_file() for path in directory.iterdir()):
            continue
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ()
        for series_id in series_ids:
            names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(directory), series_id
            )
            if not names:
                continue
            series = _inspect_series(directory, str(series_id), names)
            if series is not None:
                candidates.append(series)

    if not candidates:
        raise ValueError(f"No readable DICOM CT series was found in: {root}")
    diagnostic = [item for item in candidates if not item.is_localizer]
    ranked = diagnostic or candidates
    return sorted(
        ranked,
        key=lambda item: (len(item.files), np.prod(item.shape), item.series_id),
        reverse=True,
    )


def convert_dicom_series(
    series: DicomSeries,
    *,
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> ConvertedDicom:
    """Read a validated CT series with GDCM and cache a de-identified NIfTI."""
    report = progress or (lambda _value, _message: None)
    report(4, "Reading DICOM series")
    _validate_slice_metadata(series.files)
    cache_root = Path(cache_dir or default_dicom_cache_directory())
    digest = _series_digest(series)
    output = cache_root / f"dicom-{digest}.nii.gz"
    if output.is_file():
        return ConvertedDicom(output, series, _nifti_orientation(output), True)

    reader = sitk.ImageSeriesReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileNames([str(path) for path in series.files])
    image = reader.Execute()
    _validate_image_geometry(image, series)
    report(9, "Converting CT")

    # Rebuild the image from its HU-valued pixels so DICOM patient metadata is
    # never carried into the cache. GDCM applies rescale slope/intercept on read.
    pixels = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    sanitized = sitk.GetImageFromArray(pixels)
    sanitized.SetOrigin(image.GetOrigin())
    sanitized.SetSpacing(image.GetSpacing())
    sanitized.SetDirection(image.GetDirection())
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.nii.gz")
    try:
        sitk.WriteImage(sanitized, str(temporary), True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return ConvertedDicom(output, series, _nifti_orientation(output), False)


def _inspect_series(
    directory: Path, series_id: str, names: tuple[str, ...] | list[str]
) -> DicomSeries | None:
    files = tuple(Path(name).resolve() for name in names)
    reader = _metadata_reader(files[0])
    modality = _tag(reader, "0008|0060").upper()
    if modality != "CT":
        return None
    description = _tag(reader, "0008|103e").strip()
    rows = int(_tag(reader, "0028|0010") or reader.GetSize()[1])
    columns = int(_tag(reader, "0028|0011") or reader.GetSize()[0])
    pixel_spacing = _decimal_values(_tag(reader, "0028|0030"), expected=2)
    positions, _normal = _slice_positions(files[:2])
    slice_spacing = abs(positions[1] - positions[0]) if len(positions) > 1 else 1.0
    spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), slice_spacing)
    text = " ".join(
        (description, _tag(reader, "0008|0008"), _tag(reader, "0018|1030"))
    ).lower()
    localizer = any(word in text for word in _LOCALIZER_WORDS) or len(files) < 16
    return DicomSeries(
        directory=directory.resolve(),
        series_id=series_id,
        files=files,
        description=description,
        shape=(columns, rows, len(files)),
        spacing=spacing,
        is_localizer=localizer,
    )


def _validate_slice_metadata(files: tuple[Path, ...]) -> None:
    if len(files) < 2:
        raise ValueError("A CT series must contain at least two slices")
    reference = _metadata_reader(files[0])
    rows = _tag(reference, "0028|0010")
    columns = _tag(reference, "0028|0011")
    spacing = np.asarray(_decimal_values(_tag(reference, "0028|0030"), 2))
    orientation = np.asarray(_decimal_values(_tag(reference, "0020|0037"), 6))
    for path in files[1:]:
        current = _metadata_reader(path)
        if (_tag(current, "0028|0010"), _tag(current, "0028|0011")) != (
            rows,
            columns,
        ):
            raise ValueError("DICOM slices have inconsistent dimensions")
        if not np.allclose(
            _decimal_values(_tag(current, "0028|0030"), 2), spacing, atol=1e-5
        ):
            raise ValueError("DICOM slices have inconsistent in-plane spacing")
        if not np.allclose(
            _decimal_values(_tag(current, "0020|0037"), 6),
            orientation,
            atol=1e-5,
        ):
            raise ValueError("DICOM slices have inconsistent orientation")
    positions, _normal = _slice_positions(files)
    gaps = np.diff(sorted(positions))
    if np.any(np.abs(gaps) < 1e-5) or not np.allclose(
        np.abs(gaps), np.median(np.abs(gaps)), atol=1e-3, rtol=1e-3
    ):
        raise ValueError("DICOM slices have inconsistent slice spacing")


def _validate_image_geometry(image: sitk.Image, series: DicomSeries) -> None:
    if tuple(int(value) for value in image.GetSize()) != series.shape:
        raise ValueError("GDCM produced an unexpected CT volume size")
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("DICOM direction cosines are not orthonormal")
    if np.linalg.det(direction) < 0.99:
        raise ValueError("DICOM direction matrix is invalid")
    if any(value <= 0 or not np.isfinite(value) for value in image.GetSpacing()):
        raise ValueError("DICOM spacing is invalid")


def _slice_positions(files: tuple[Path, ...]) -> tuple[list[float], np.ndarray]:
    first = _metadata_reader(files[0])
    orientation = np.asarray(_decimal_values(_tag(first, "0020|0037"), 6))
    normal = np.cross(orientation[:3], orientation[3:])
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-4):
        raise ValueError("DICOM ImageOrientationPatient is invalid")
    positions = []
    for path in files:
        reader = _metadata_reader(path)
        point = np.asarray(_decimal_values(_tag(reader, "0020|0032"), 3))
        positions.append(float(np.dot(point, normal)))
    return positions, normal


def _metadata_reader(path: Path) -> sitk.ImageFileReader:
    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    return reader


def _tag(reader: sitk.ImageFileReader, key: str) -> str:
    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""


def _decimal_values(value: str, expected: int) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split("\\"))
    except ValueError as error:
        raise ValueError("DICOM geometry metadata is invalid") from error
    if len(result) != expected or not all(np.isfinite(result)):
        raise ValueError("DICOM geometry metadata is missing or invalid")
    return result


def _series_digest(series: DicomSeries) -> str:
    state = [series.series_id]
    for path in series.files:
        stat = path.stat()
        state.extend((str(path), str(stat.st_size), str(stat.st_mtime_ns)))
    return hashlib.sha256("|".join(state).encode("utf-8")).hexdigest()


def _nifti_orientation(path: Path) -> str:
    return "".join(nib.aff2axcodes(nib.load(str(path)).affine))
