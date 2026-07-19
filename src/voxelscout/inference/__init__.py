"""Replaceable vertebra-segmentation backends."""

from voxelscout.inference.backend import (
    InferenceUnavailableError,
    SegmentationBackend,
)
from voxelscout.inference.nnunet_backend import NnUNetBackend, NnUNetConfig

__all__ = [
    "InferenceUnavailableError",
    "NnUNetBackend",
    "NnUNetConfig",
    "SegmentationBackend",
]
