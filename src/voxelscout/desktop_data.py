"""Header-only CT loading and cached per-vertebra meshes for the desktop app."""

from __future__ import annotations

import threading
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from voxelscout.anatomy import vertebra_info


ProgressCallback = Callable[[int, str], None]
Bounds3D = tuple[slice, slice, slice]


REGION_COLOURS = {
    "Cervical spine": "#22b8a7",
    "Thoracic spine": "#3986d7",
    "Lumbar spine": "#e58a35",
    "Unmapped region": "#9b7ed0",
}
MIN_LABEL_VOXELS = 50


@dataclass(frozen=True)
class VertebraMesh:
    label: int
    vertices: np.ndarray
    faces: np.ndarray
    colour: str


@dataclass(frozen=True)
class CTVolume:
    data: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    orientation: str
    source_path: Path


@dataclass(frozen=True)
class SegmentationVolume:
    labels: np.ndarray
    affine: np.ndarray
    source: str
    source_path: Path | None = None


@dataclass(frozen=True)
class SegmentedCase:
    ct_path: Path
    mask_path: Path
    name: str
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    orientation: str
    meshes: tuple[VertebraMesh, ...]

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(mesh.label for mesh in self.meshes)

    @property
    def mesh_memory_mib(self) -> float:
        return sum(
            mesh.vertices.nbytes + mesh.faces.nbytes for mesh in self.meshes
        ) / 2**20


_CASE_CACHE: dict[tuple[object, ...], SegmentedCase] = {}
_CACHE_LOCK = threading.Lock()


def find_companion_mask(ct_path: Path) -> Path | None:
    """Find a VerSe mask next to a CT or under the derivatives directory."""
    ct_path = Path(ct_path).resolve()
    name = ct_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else ct_path.stem
    base = stem[:-3] if stem.lower().endswith("_ct") else stem
    for filename in (
        f"{base}_seg-vert_msk.nii.gz",
        f"{base}_mask.nii.gz",
        f"{base}_seg.nii.gz",
    ):
        candidate = ct_path.parent / filename
        if candidate.is_file():
            return candidate

    raw_root = next(
        (parent for parent in ct_path.parents if parent.name.lower() == "rawdata"),
        None,
    )
    if raw_root is None:
        return None
    relative = ct_path.relative_to(raw_root).parts
    if len(relative) < 2:
        return None
    candidates = sorted(
        (raw_root.parent / "derivatives" / relative[0]).glob(
            "*_seg-vert_msk.nii.gz"
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _cache_key(
    ct_path: Path,
    mask_path: Path,
    sample_step: int,
    target_faces: int,
) -> tuple[object, ...]:
    ct_stat = ct_path.stat()
    mask_stat = mask_path.stat()
    return (
        str(ct_path.resolve()),
        ct_stat.st_mtime_ns,
        ct_stat.st_size,
        str(mask_path.resolve()),
        mask_stat.st_mtime_ns,
        mask_stat.st_size,
        int(sample_step),
        int(target_faces),
    )


def _array_digest(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _build_cache_key(
    ct: CTVolume,
    segmentation: SegmentationVolume,
    sample_step: int,
    target_faces: int,
) -> tuple[object, ...]:
    ct_path = Path(ct.source_path).resolve()
    ct_stat = ct_path.stat()
    if segmentation.source_path is not None:
        segmentation_path = Path(segmentation.source_path).resolve()
        segmentation_stat = segmentation_path.stat()
        segmentation_identity: tuple[object, ...] = (
            str(segmentation_path),
            segmentation_stat.st_mtime_ns,
            segmentation_stat.st_size,
        )
    else:
        segmentation_identity = (_array_digest(segmentation.labels),)
    return (
        str(ct_path),
        ct_stat.st_mtime_ns,
        ct_stat.st_size,
        segmentation.source,
        *segmentation_identity,
        int(sample_step),
        int(target_faces),
    )


def _compact_mask(data: np.ndarray) -> np.ndarray:
    array = np.asanyarray(data)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array)
    minimum = int(array.min()) if array.size else 0
    maximum = int(array.max()) if array.size else 0
    dtype = np.uint8 if minimum >= 0 and maximum <= 255 else np.int16
    return np.asarray(array, dtype=dtype)


def _label_bounds(mask: np.ndarray) -> dict[int, Bounds3D]:
    coordinates = np.nonzero(mask)
    if not len(coordinates[0]):
        return {}
    foreground_labels = mask[coordinates]
    labels = tuple(int(value) for value in np.unique(foreground_labels))
    result: dict[int, Bounds3D] = {}
    for label in labels:
        selected = foreground_labels == label
        if int(np.count_nonzero(selected)) < MIN_LABEL_VOXELS:
            continue
        result[label] = tuple(
            slice(
                int(coordinates[axis][selected].min()),
                int(coordinates[axis][selected].max()) + 1,
            )
            for axis in range(3)
        )
    return result


def _simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use VTK topology-preserving decimation only when a mesh is oversized."""
    if len(faces) <= target_faces:
        return vertices.astype(np.float32), faces.astype(np.int32)

    import pyvista as pv

    vtk_faces = np.column_stack(
        (np.full(len(faces), 3, dtype=np.int32), faces.astype(np.int32))
    ).ravel()
    poly = pv.PolyData(vertices, vtk_faces)
    reduction = min(0.92, max(0.0, 1.0 - target_faces / len(faces)))
    simplified = poly.decimate_pro(reduction, preserve_topology=True)
    triangles = simplified.faces.reshape(-1, 4)[:, 1:]
    return (
        np.asarray(simplified.points, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
    )


def _mesh_for_label(
    mask: np.ndarray,
    label: int,
    bounds: Bounds3D,
    spacing: tuple[float, float, float],
    sample_step: int,
    target_faces: int,
) -> VertebraMesh:
    from skimage.measure import marching_cubes

    binary = np.pad(mask[bounds] == label, 1)
    binary = binary.astype(np.uint8)
    step_size = max(1, int(sample_step))
    try:
        vertices, faces, _normals, _values = marching_cubes(
            binary,
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
    except (RuntimeError, ValueError):
        if step_size == 1:
            raise
        vertices, faces, _normals, _values = marching_cubes(
            binary,
            level=0.5,
            spacing=spacing,
            step_size=1,
            allow_degenerate=False,
        )
    if not len(vertices) or not len(faces):
        raise ValueError(f"Label {label} did not produce a valid surface")
    offset = (
        np.asarray([axis.start or 0 for axis in bounds], dtype=float) - 1.0
    ) * np.asarray(spacing)
    vertices += offset
    vertices, faces = _simplify_mesh(vertices, faces, target_faces)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    info = vertebra_info(label)
    return VertebraMesh(
        label=label,
        vertices=vertices,
        faces=faces,
        colour=REGION_COLOURS.get(info.region, REGION_COLOURS["Unmapped region"]),
    )


def load_ct_volume(
    ct_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> CTVolume:
    """Load a three-dimensional NIfTI CT without resampling or normalization."""
    report = progress or (lambda _value, _message: None)
    ct_path = Path(ct_path).resolve()
    report(4, "Reading CT")
    image = nib.load(str(ct_path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape {image.shape}")
    data = np.asanyarray(image.dataobj)
    return CTVolume(
        data=data,
        affine=np.asarray(image.affine, dtype=float),
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(image.affine)),
        orientation="".join(nib.aff2axcodes(image.affine)),
        source_path=ct_path,
    )


def load_segmentation_volume(
    mask_path: Path,
    *,
    source: str = "trusted-mask",
) -> SegmentationVolume:
    """Load a labelled NIfTI segmentation without changing label identities."""
    mask_path = Path(mask_path).resolve()
    image = nib.load(str(mask_path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D segmentation, got shape {image.shape}")
    return SegmentationVolume(
        labels=_compact_mask(np.asanyarray(image.dataobj)),
        affine=np.asarray(image.affine, dtype=float),
        source=source,
        source_path=mask_path,
    )


def build_segmented_case(
    ct: CTVolume,
    segmentation: SegmentationVolume,
    *,
    sample_step: int = 2,
    target_faces_per_vertebra: int = 12_000,
    progress: ProgressCallback | None = None,
) -> SegmentedCase:
    """Validate in-memory volumes and build one cached mesh per vertebra label."""
    report = progress or (lambda _value, _message: None)
    ct_shape = tuple(int(value) for value in np.asanyarray(ct.data).shape)
    segmentation_shape = tuple(
        int(value) for value in np.asanyarray(segmentation.labels).shape
    )
    report(10, "Validating prediction")
    if ct_shape != segmentation_shape:
        raise ValueError(
            f"CT shape {ct_shape} does not match mask shape {segmentation_shape}"
        )
    if not np.allclose(ct.affine, segmentation.affine, atol=1e-3, rtol=1e-5):
        raise ValueError("CT and segmentation do not use the same spatial coordinates")

    key = _build_cache_key(
        ct, segmentation, sample_step, target_faces_per_vertebra
    )
    with _CACHE_LOCK:
        cached = _CASE_CACHE.get(key)
    if cached is not None:
        report(100, "Loaded cached 3D spine")
        return cached

    # Canonicalise geometry only for display; no resampling or normalization occurs.
    mask_image = nib.Nifti1Image(
        _compact_mask(segmentation.labels), np.asarray(segmentation.affine)
    )
    canonical_mask = nib.as_closest_canonical(mask_image)
    mask = _compact_mask(np.asanyarray(canonical_mask.dataobj))
    spacing = tuple(
        float(value) for value in nib.affines.voxel_sizes(canonical_mask.affine)
    )
    orientation = "".join(nib.aff2axcodes(canonical_mask.affine))
    bounds = _label_bounds(mask)
    if not bounds:
        raise ValueError("The segmentation mask contains no vertebra labels")

    meshes: list[VertebraMesh] = []
    labels = tuple(sorted(bounds))
    for index, label in enumerate(labels, start=1):
        report(
            12 + int(82 * (index - 1) / len(labels)),
            f"Building {vertebra_info(label).code} mesh ({index}/{len(labels)})",
        )
        try:
            mesh = _mesh_for_label(
                mask,
                label,
                bounds[label],
                spacing,
                sample_step,
                target_faces_per_vertebra,
            )
        except (RuntimeError, ValueError):
            continue
        meshes.append(mesh)

    report(96, "Finalising 3D model")
    result = SegmentedCase(
        ct_path=Path(ct.source_path).resolve(),
        mask_path=(
            Path(segmentation.source_path).resolve()
            if segmentation.source_path is not None
            else Path(ct.source_path).resolve()
        ),
        name=Path(ct.source_path).parent.name or Path(ct.source_path).name,
        shape=tuple(int(value) for value in canonical_mask.shape),
        spacing=spacing,
        orientation=orientation,
        meshes=tuple(meshes),
    )
    with _CACHE_LOCK:
        _CASE_CACHE[key] = result
    report(100, "3D spine ready")
    return result


def load_segmented_case(
    ct_path: Path,
    mask_path: Path,
    *,
    sample_step: int = 2,
    target_faces_per_vertebra: int = 12_000,
    progress: ProgressCallback | None = None,
) -> SegmentedCase:
    """Compatibility wrapper for a trusted CT and segmentation file pair."""
    ct = load_ct_volume(ct_path, progress=progress)
    segmentation = load_segmentation_volume(mask_path)
    return build_segmented_case(
        ct,
        segmentation,
        sample_step=sample_step,
        target_faces_per_vertebra=target_faces_per_vertebra,
        progress=progress,
    )
