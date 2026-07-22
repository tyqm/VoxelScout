"""Model-agnostic metrics for an optional reference segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: float
    iou: float
    hd95_mm: float


def evaluate_segmentation(
    prediction: np.ndarray, reference: np.ndarray, affine: np.ndarray
) -> SegmentationMetrics:
    """Calculate macro per-label Dice/IoU and surface HD95 in millimetres."""
    predicted_labels = np.asarray(prediction)
    expected_labels = np.asarray(reference)
    predicted = predicted_labels > 0
    expected = expected_labels > 0
    if predicted.shape != expected.shape:
        raise ValueError("Prediction and ground truth shapes do not match")
    labels = sorted(
        (set(np.unique(predicted_labels)) | set(np.unique(expected_labels))) - {0}
    )
    if not labels:
        return SegmentationMetrics(dice=1.0, iou=1.0, hd95_mm=0.0)
    dice_values: list[float] = []
    iou_values: list[float] = []
    hd95_values: list[float] = []
    for label in labels:
        predicted = predicted_labels == label
        expected = expected_labels == label
        intersection = int(np.count_nonzero(predicted & expected))
        predicted_count = int(np.count_nonzero(predicted))
        expected_count = int(np.count_nonzero(expected))
        union = predicted_count + expected_count - intersection
        dice_values.append(2 * intersection / (predicted_count + expected_count))
        iou_values.append(intersection / union)
        hd95_values.append(_surface_hd95(predicted, expected, np.asarray(affine, dtype=float)))
    return SegmentationMetrics(
        dice=float(np.mean(dice_values)),
        iou=float(np.mean(iou_values)),
        hd95_mm=float(np.mean(hd95_values)),
    )


def _surface_hd95(predicted: np.ndarray, expected: np.ndarray, affine: np.ndarray) -> float:
    if not np.any(predicted) or not np.any(expected):
        return 0.0 if np.array_equal(predicted, expected) else float("inf")
    first = _world_points(np.argwhere(predicted & ~binary_erosion(predicted)), affine)
    second = _world_points(np.argwhere(expected & ~binary_erosion(expected)), affine)
    distances = np.concatenate(
        (cKDTree(second).query(first, workers=1)[0], cKDTree(first).query(second, workers=1)[0])
    )
    return float(np.percentile(distances, 95))


def _world_points(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((indices, np.ones(len(indices))))
    return (homogeneous @ affine.T)[:, :3]
