"""Backend contract shared by the desktop worker and inference implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from voxelscout.desktop_data import ProgressCallback, SegmentationVolume


class InferenceUnavailableError(RuntimeError):
    """Raised when automatic inference is not installed or configured."""


class SegmentationBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def cache_key(self) -> str: ...

    def predict(
        self,
        ct_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> SegmentationVolume: ...
