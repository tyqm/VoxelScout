from __future__ import annotations

import numpy as np

from voxelscout.ct_overlay import blend_vertebra_overlay, label_centres


def test_label_centres_use_each_labels_axial_extent() -> None:
    mask = np.zeros((8, 8, 12), dtype=np.uint8)
    mask[1:4, 1:4, 2:6] = 20
    mask[4:7, 4:7, 7:11] = 21
    assert label_centres(mask) == {20: 4, 21: 8}


def test_selected_overlay_is_stronger_than_context() -> None:
    image = np.full((12, 12), 80, dtype=np.uint8)
    labels = np.zeros_like(image)
    labels[2:6, 2:6] = 20
    labels[7:11, 7:11] = 21
    result = blend_vertebra_overlay(image, labels, 20, 0.6)
    selected_change = np.abs(result[3, 3].astype(int) - 80).sum()
    context_change = np.abs(result[8, 8].astype(int) - 80).sum()
    assert result.shape == (12, 12, 3)
    assert selected_change > context_change


def test_disabled_overlay_preserves_grayscale_values() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    labels = np.ones((5, 5), dtype=np.uint8)
    result = blend_vertebra_overlay(image, labels, 1, 0.0)
    assert np.array_equal(result, np.repeat(image[..., None], 3, axis=2))
