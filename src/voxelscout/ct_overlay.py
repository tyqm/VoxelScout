"""Display-only helpers linking labelled vertebra masks to axial CT slices."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries


def label_centres(mask: np.ndarray) -> dict[int, int]:
    """Return the centre axial index of each foreground label."""
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return {}
    foreground_labels = mask[coordinates]
    result: dict[int, int] = {}
    for value in np.unique(foreground_labels):
        label = int(value)
        if label <= 0:
            continue
        axial = coordinates[2][foreground_labels == value]
        if axial.size:
            result[label] = int(round((axial.min() + axial.max()) / 2))
    return result


def blend_vertebra_overlay(
    grayscale: np.ndarray,
    labels: np.ndarray,
    selected_label: int | None,
    opacity: float,
) -> np.ndarray:
    """Blend selected and contextual vertebra labels into an RGB display image."""
    base = np.asarray(grayscale, dtype=np.uint8)
    if base.ndim != 2 or labels.shape != base.shape:
        raise ValueError("Overlay image and label slice must have matching 2D shapes")
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    strength = float(np.clip(opacity, 0.0, 1.0))
    foreground = labels > 0
    if not np.any(foreground) or strength <= 0:
        return rgb.astype(np.uint8)

    selected = foreground if selected_label is None else labels == int(selected_label)
    context = foreground & ~selected
    _blend(rgb, context, np.array([105, 137, 160]), strength * 0.12)
    _blend(rgb, find_boundaries(context, mode="inner"), np.array([150, 176, 194]), strength * 0.28)
    _blend(rgb, selected, np.array([244, 201, 93]), strength * 0.34)
    _blend(
        rgb,
        find_boundaries(selected, mode="inner"),
        np.array([255, 221, 112]),
        min(0.95, strength + 0.25),
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _blend(image: np.ndarray, selected: np.ndarray, colour: np.ndarray, alpha: float) -> None:
    if np.any(selected) and alpha > 0:
        image[selected] = image[selected] * (1.0 - alpha) + colour * alpha
