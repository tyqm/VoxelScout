import numpy as np
import pytest

from voxelscout.inference.metrics import evaluate_segmentation


def test_identical_masks_have_perfect_metrics() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 20

    result = evaluate_segmentation(mask, mask, np.eye(4))

    assert result.dice == pytest.approx(1.0)
    assert result.iou == pytest.approx(1.0)
    assert result.hd95_mm == pytest.approx(0.0)


def test_hd95_uses_physical_geometry() -> None:
    first = np.zeros((8, 8, 8), dtype=np.uint8)
    second = np.zeros_like(first)
    first[2:5, 2:5, 2:5] = 1
    second[3:6, 2:5, 2:5] = 1
    affine = np.diag([2.0, 1.0, 1.0, 1.0])

    result = evaluate_segmentation(first, second, affine)

    assert result.dice < 1.0
    assert result.iou < 1.0
    assert result.hd95_mm == pytest.approx(2.0)


def test_wrong_vertebra_identity_is_not_counted_as_correct_foreground() -> None:
    prediction = np.zeros((6, 6, 6), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[1:5, 1:5, 1:5] = 20
    reference[1:5, 1:5, 1:5] = 21

    result = evaluate_segmentation(prediction, reference, np.eye(4))

    assert result.dice == 0.0
    assert result.iou == 0.0
    assert np.isinf(result.hd95_mm)
