"""Header-only CT loading and cached per-vertebra meshes for the desktop app."""

from __future__ import annotations

import threading
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


@dataclass(frozen=True)
class VertebraMesh:
    label: int
    vertices: np.ndarray
    faces: np.ndarray
    colour: str


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
    vertices, faces, _normals, _values = marching_cubes(
        binary.astype(np.uint8),
        level=0.5,
        spacing=spacing,
        step_size=max(1, int(sample_step)),
        allow_degenerate=False,
    )
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


def load_segmented_case(
    ct_path: Path,
    mask_path: Path,
    *,
    sample_step: int = 2,
    target_faces_per_vertebra: int = 12_000,
    progress: ProgressCallback | None = None,
) -> SegmentedCase:
    """Read only the CT header and build one cached mesh per mask label."""
    report = progress or (lambda _value, _message: None)
    ct_path = Path(ct_path).resolve()
    mask_path = Path(mask_path).resolve()
    key = _cache_key(ct_path, mask_path, sample_step, target_faces_per_vertebra)
    with _CACHE_LOCK:
        cached = _CASE_CACHE.get(key)
    if cached is not None:
        report(100, "Loaded cached 3D spine")
        return cached

    report(4, "Reading image headers")
    ct_image = nib.load(str(ct_path))
    mask_image = nib.load(str(mask_path))
    if ct_image.shape != mask_image.shape:
        raise ValueError(
            f"CT shape {ct_image.shape} does not match mask shape {mask_image.shape}"
        )
    if not np.allclose(ct_image.affine, mask_image.affine, atol=1e-3):
        raise ValueError("CT and segmentation do not use the same spatial coordinates")

    # Canonicalise the compact mask. The CT voxel array is deliberately never read.
    report(10, "Reading segmentation mask")
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
        meshes.append(
            _mesh_for_label(
                mask,
                label,
                bounds[label],
                spacing,
                sample_step,
                target_faces_per_vertebra,
            )
        )

    report(96, "Finalising 3D model")
    result = SegmentedCase(
        ct_path=ct_path,
        mask_path=mask_path,
        name=ct_path.parent.name or ct_path.name,
        shape=tuple(int(value) for value in canonical_mask.shape),
        spacing=spacing,
        orientation=orientation,
        meshes=tuple(meshes),
    )
    with _CACHE_LOCK:
        _CASE_CACHE[key] = result
    report(100, "3D spine ready")
    return result
