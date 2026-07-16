"""Core rendering and geometry helpers for the VoxelScout GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from voxelscout.anatomy import vertebra_info


PALETTE = np.asarray(
    [
        (230, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (26, 188, 156),
        (231, 76, 120),
        (127, 140, 141),
        (52, 73, 94),
    ],
    dtype=np.float32,
)


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume in canonical RAS orientation."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    return data, image.affine, spacing


def validate_mask(
    image: np.ndarray,
    image_affine: np.ndarray,
    mask: np.ndarray,
    mask_affine: np.ndarray,
) -> None:
    """Validate that an optional mask belongs to the displayed CT."""
    if image.shape != mask.shape:
        raise ValueError(f"CT shape {image.shape} does not match mask shape {mask.shape}")
    if not np.allclose(image_affine, mask_affine, atol=1e-3):
        raise ValueError("CT and mask use different spatial coordinates")


def window_ct(volume: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Apply a CT display window and return values in [0, 1]."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    lower = centre - width / 2
    return np.clip((volume - lower) / width, 0.0, 1.0)


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract and rotate one anatomical slice for screen display."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"slice {index} outside axis {axis}")
    return np.rot90(np.take(volume, index, axis=axis))


def render_slice(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    axis: int,
    index: int,
    centre: float,
    width: float,
    opacity: float,
) -> np.ndarray:
    """Render one CT slice as an RGB image with an optional coloured mask."""
    grey = extract_slice(window_ct(image, centre, width), axis, index)
    rgb = np.repeat(grey[..., None], 3, axis=-1) * 255.0

    if mask is not None and opacity > 0:
        label_slice = extract_slice(mask, axis, index).astype(np.int32)
        for label in np.unique(label_slice):
            if label <= 0:
                continue
            selected = label_slice == label
            colour = PALETTE[(label - 1) % len(PALETTE)]
            rgb[selected] = (
                rgb[selected] * (1.0 - opacity) + colour * opacity
            )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def visible_labels(mask: np.ndarray | None) -> list[int]:
    """Return positive integer labels available for navigation."""
    if mask is None:
        return []
    return [int(value) for value in np.unique(mask) if value > 0]


def label_options(mask: np.ndarray | None) -> dict[str, int]:
    """Create user-facing select-box labels."""
    return {
        f"{vertebra_info(label).code} — {vertebra_info(label).plain_location}": label
        for label in visible_labels(mask)
    }


def label_centroid(mask: np.ndarray, label: int) -> tuple[int, int, int]:
    """Find the nearest integer centre of a vertebra label."""
    points = np.argwhere(mask == label)
    if points.size == 0:
        raise ValueError(f"Label {label} is not present")
    return tuple(int(round(value)) for value in points.mean(axis=0))


def surface_from_mask(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    label: int | None = None,
    step_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a lightweight surface mesh for the whole mask or one vertebra."""
    binary = mask == label if label is not None else mask > 0
    if not np.any(binary):
        raise ValueError("No labelled spine voxels are available for 3D view")
    vertices, faces, _, _ = marching_cubes(
        binary.astype(np.uint8),
        level=0.5,
        spacing=tuple(float(value) for value in spacing),
        step_size=max(1, int(step_size)),
        allow_degenerate=False,
    )
    return vertices, faces
