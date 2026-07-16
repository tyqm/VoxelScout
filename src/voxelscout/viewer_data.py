"""Data loading and geometry helpers for the VoxelScout desktop viewer.

The functions in this module deliberately do not import Tk.  Keeping the data
layer separate makes the loading, label matching, and surface generation usable
from tests and from future non-desktop front ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from voxelscout.inspect_case import validate_pair


ProgressCallback = Callable[[str], None]


REGION_COLOURS = {
    "Cervical": "#2dd4bf",
    "Thoracic": "#38bdf8",
    "Lumbar": "#fb923c",
    "Other": "#a78bfa",
}


@dataclass(frozen=True)
class LabelGeometry:
    """Location information for one labelled vertebra."""

    label: int
    bounds: tuple[slice, slice, slice]
    centre: tuple[float, float, float]


@dataclass
class VolumeCase:
    """A canonical RAS CT volume and an optional aligned vertebra mask."""

    source_path: Path
    image: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    orientation: str
    mask: np.ndarray | None = None
    label_path: Path | None = None
    label_geometry: dict[int, LabelGeometry] = field(default_factory=dict)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(self.label_geometry))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.image.shape)


@dataclass(frozen=True)
class SurfaceMesh:
    """A surface ready to pass to Matplotlib's ``plot_trisurf``."""

    label: int | None
    vertices_mm: np.ndarray
    faces: np.ndarray


def vertebra_name(label: int) -> str:
    """Translate VerSe integer IDs into familiar vertebral names."""
    if 1 <= label <= 7:
        return f"C{label}"
    if 8 <= label <= 19:
        return f"T{label - 7}"
    if 20 <= label <= 25:
        return f"L{label - 19}"
    special = {26: "Sacrum", 27: "Coccyx", 28: "T13"}
    return special.get(label, f"Label {label}")


def vertebra_region(label: int) -> str:
    """Return the broad spinal region represented by a VerSe label."""
    if 1 <= label <= 7:
        return "Cervical"
    if 8 <= label <= 19 or label == 28:
        return "Thoracic"
    if 20 <= label <= 25:
        return "Lumbar"
    return "Other"


def is_nifti(path: Path) -> bool:
    """Return whether *path* has a supported NIfTI filename."""
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def find_companion_label(image_path: Path) -> Path | None:
    """Find an aligned VerSe-style label file near an imported NIfTI CT.

    Both a flat case directory and the BIDS-like VerSe ``rawdata`` /
    ``derivatives`` layout are supported.
    """
    image_path = image_path.resolve()
    stem = _nifti_stem(image_path)
    base = stem[:-3] if stem.lower().endswith("_ct") else stem

    local_names = (
        f"{base}_seg-vert_msk.nii.gz",
        f"{base}_seg.nii.gz",
        f"{base}_mask.nii.gz",
        f"{base}_seg-vert_msk.nii",
        f"{base}_seg.nii",
        f"{base}_mask.nii",
    )
    for name in local_names:
        candidate = image_path.parent / name
        if candidate.is_file() and candidate != image_path:
            return candidate

    raw_root = next(
        (parent for parent in image_path.parents if parent.name.lower() == "rawdata"),
        None,
    )
    if raw_root is None:
        return None

    relative_parts = image_path.relative_to(raw_root).parts
    if len(relative_parts) < 2:
        return None
    subject = relative_parts[0]
    derivatives = raw_root.parent / "derivatives" / subject
    if not derivatives.is_dir():
        return None

    exact_names = (
        f"{base}_seg-vert_msk.nii.gz",
        f"{base}_seg-vert_msk.nii",
    )
    for name in exact_names:
        candidate = derivatives / name
        if candidate.is_file():
            return candidate

    candidates = sorted(derivatives.glob("*_seg-vert_msk.nii*"))
    return candidates[0] if len(candidates) == 1 else None


def _load_nifti(path: Path) -> nib.Nifti1Image:
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    if not is_nifti(path):
        raise ValueError(f"Unsupported image type: {path.name}")
    return nib.as_closest_canonical(nib.load(str(path)))


def _load_dicom(directory: Path) -> nib.Nifti1Image:
    """Read the largest DICOM series in a directory and convert LPS to RAS."""
    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "DICOM support requires SimpleITK. Install the project dependencies first."
        ) from error

    if not directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {directory}")

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ()
    series = []
    for series_id in series_ids:
        files = tuple(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(directory), series_id
            )
        )
        if files:
            series.append(files)
    if not series:
        raise ValueError(f"No readable DICOM image series found in: {directory}")

    files = max(series, key=len)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()

    # SimpleITK returns array axes as z, y, x and physical coordinates in LPS.
    data = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = np.asarray(image.GetOrigin(), dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return nib.as_closest_canonical(
        nib.Nifti1Image(data.astype(np.float32, copy=False), lps_to_ras @ affine_lps)
    )


def compute_label_geometry(mask: np.ndarray) -> dict[int, LabelGeometry]:
    """Compute compact bounding boxes and centres with one full-volume scan."""
    from scipy import ndimage

    labels = tuple(int(value) for value in np.unique(mask) if int(value) != 0)
    if not labels:
        return {}

    objects = ndimage.find_objects(mask)
    result: dict[int, LabelGeometry] = {}
    for label in labels:
        bounds = objects[label - 1] if label <= len(objects) else None
        if bounds is None:
            continue
        local = np.argwhere(mask[bounds] == label)
        starts = np.asarray([axis.start or 0 for axis in bounds], dtype=float)
        centre = tuple(float(value) for value in (local.mean(axis=0) + starts))
        result[label] = LabelGeometry(label, bounds, centre)
    return result


def load_case(
    source_path: Path,
    label_path: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> VolumeCase:
    """Load a NIfTI file or DICOM directory with an optional vertebra mask."""
    source_path = Path(source_path).resolve()
    report = progress or (lambda _message: None)
    report("Reading CT volume…")
    ct_image = _load_dicom(source_path) if source_path.is_dir() else _load_nifti(source_path)

    selected_label = Path(label_path).resolve() if label_path is not None else None
    if selected_label is None and source_path.is_file():
        selected_label = find_companion_label(source_path)

    mask: np.ndarray | None = None
    geometry: dict[int, LabelGeometry] = {}
    if selected_label is not None:
        report("Checking vertebra labels…")
        label_image = _load_nifti(selected_label)
        validate_pair(ct_image, label_image)
        mask = np.asanyarray(label_image.dataobj).astype(np.int16, copy=False)
        geometry = compute_label_geometry(mask)

    report("Preparing display data…")
    image = ct_image.get_fdata(dtype=np.float32)
    spacing = tuple(
        float(value) for value in nib.affines.voxel_sizes(ct_image.affine)
    )
    return VolumeCase(
        source_path=source_path,
        image=image,
        affine=np.asarray(ct_image.affine),
        spacing_mm=spacing,
        orientation="".join(nib.aff2axcodes(ct_image.affine)),
        mask=mask,
        label_path=selected_label,
        label_geometry=geometry,
    )


def default_slice_indices(case: VolumeCase) -> tuple[int, int, int]:
    """Choose a useful initial crosshair, preferring the labelled anatomy."""
    if case.label_geometry:
        centres = np.asarray(
            [item.centre for item in case.label_geometry.values()], dtype=float
        )
        return tuple(int(round(value)) for value in centres.mean(axis=0))
    return tuple(size // 2 for size in case.shape)


def orthogonal_slice(
    volume: np.ndarray, axis: int, index: int
) -> np.ndarray:
    """Extract and rotate a canonical sagittal, coronal, or axial slice."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"slice {index} is outside axis {axis}")
    return np.rot90(np.take(volume, index, axis=axis))


def build_surface_meshes(
    case: VolumeCase,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[SurfaceMesh, ...]:
    """Build lightweight labelled surfaces, or a bone preview without labels."""
    from skimage import measure

    report = progress or (lambda _message: None)
    meshes: list[SurfaceMesh] = []

    if case.mask is not None and case.label_geometry:
        count = len(case.labels)
        for number, label in enumerate(case.labels, start=1):
            report(
                f"Building 3D vertebrae… {number}/{count} ({vertebra_name(label)})"
            )
            geometry = case.label_geometry[label]
            local = np.pad(case.mask[geometry.bounds] == label, 1)
            step_size = max(1, int(np.ceil(max(local.shape) / 96)))
            vertices, faces, _normals, _values = measure.marching_cubes(
                local.astype(np.uint8),
                level=0.5,
                spacing=case.spacing_mm,
                step_size=step_size,
                allow_degenerate=False,
            )
            # Padding supplies a background border even when the foreground fills
            # its tight bounding box. Subtract one voxel to preserve its position.
            offset = (
                np.asarray(
                    [axis.start or 0 for axis in geometry.bounds], dtype=float
                )
                - 1.0
            ) * np.asarray(case.spacing_mm)
            meshes.append(SurfaceMesh(label, vertices + offset, faces))
        return tuple(meshes)

    report("Building a low-resolution 3D bone preview…")
    stride = max(2, int(np.ceil(max(case.shape) / 144)))
    sampled = case.image[::stride, ::stride, ::stride]
    level = 250.0
    if not float(sampled.min()) < level < float(sampled.max()):
        return ()
    vertices, faces, _normals, _values = measure.marching_cubes(
        sampled,
        level=level,
        spacing=tuple(value * stride for value in case.spacing_mm),
        step_size=1,
        allow_degenerate=False,
    )
    return (SurfaceMesh(None, vertices, faces),)
