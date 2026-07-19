"""Shared VerSe/nnU-Net label definitions and reversible mappings."""

from __future__ import annotations

import numpy as np


SUPPORTED_VERSE_LABELS = frozenset((*range(26), 28))
SUPPORTED_TRAINING_LABELS = frozenset(range(27))
VERSE_TO_TRAINING = {**{value: value for value in range(26)}, 28: 26}
TRAINING_TO_VERSE = {**{value: value for value in range(26)}, 26: 28}


def _integer_labels(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asanyarray(data)
    unique = np.unique(values)
    if not np.all(np.isfinite(unique)):
        raise ValueError("mask contains non-finite values")
    rounded = np.rint(unique)
    if not np.allclose(unique, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("mask contains non-integer label values")
    return values, tuple(sorted(int(value) for value in rounded))


def verse_to_training_labels(data: np.ndarray) -> np.ndarray:
    values, labels = _integer_labels(data)
    unsupported = sorted(set(labels) - SUPPORTED_VERSE_LABELS)
    if unsupported:
        raise ValueError(f"unsupported VerSe labels: {unsupported}")
    converted = values.astype(np.uint8, copy=False)
    if 28 in labels:
        converted = converted.copy()
        converted[values == 28] = 26
    return converted


def training_to_verse_labels(data: np.ndarray) -> np.ndarray:
    values, labels = _integer_labels(data)
    unsupported = sorted(set(labels) - SUPPORTED_TRAINING_LABELS)
    if unsupported:
        raise ValueError(f"unsupported prediction labels: {unsupported}")
    converted = values.astype(np.uint8, copy=False)
    if 26 in labels:
        converted = converted.copy()
        converted[values == 26] = 28
    return converted
