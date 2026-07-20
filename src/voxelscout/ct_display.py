"""Pure, display-only CT inspection and grayscale transformations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from skimage import exposure


@dataclass(frozen=True)
class HUReport:
    finite_min: float
    finite_max: float
    nonfinite_count: int
    below_clip_count: int
    above_clip_count: int


@dataclass(frozen=True)
class DisplaySettings:
    window_center: float
    window_width: float
    clip_enabled: bool = True
    clip_min: float = -1024.0
    clip_max: float = 3071.0
    transform: str = "None"
    gamma: float = 0.8
    sigmoid_gain: float = 8.0
    sigmoid_cutoff: float = 0.5
    clahe_enabled: bool = False
    clahe_clip_limit: float = 0.01

    def before_settings(self) -> "DisplaySettings":
        return replace(
            self,
            clip_enabled=False,
            transform="None",
            clahe_enabled=False,
        )


def inspect_hu(
    data: np.ndarray,
    *,
    clip_min: float = -1024.0,
    clip_max: float = 3071.0,
) -> HUReport:
    values = np.asanyarray(data)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("CT contains no finite HU values")
    valid = values[finite]
    return HUReport(
        finite_min=float(np.min(valid)),
        finite_max=float(np.max(valid)),
        nonfinite_count=int(values.size - np.count_nonzero(finite)),
        below_clip_count=int(np.count_nonzero(valid < clip_min)),
        above_clip_count=int(np.count_nonzero(valid > clip_max)),
    )


def automatic_window(
    data: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    clip_min: float = -1024.0,
    clip_max: float = 3071.0,
    maximum_samples: int = 1_000_000,
) -> tuple[float, float]:
    """Return deterministic robust center/width from finite, clipped HU samples."""
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Automatic-window percentiles are invalid")
    flat = np.asanyarray(data).reshape(-1)
    if flat.size > maximum_samples:
        step = int(np.ceil(flat.size / maximum_samples))
        flat = flat[::step]
    finite = flat[np.isfinite(flat)].astype(np.float32, copy=False)
    if not finite.size:
        raise ValueError("CT contains no finite HU values")
    clipped = np.clip(finite, clip_min, clip_max)
    low, high = np.percentile(clipped, (lower_percentile, upper_percentile))
    width = max(float(high - low), 1.0)
    return float((high + low) / 2.0), width


def render_ct_slice(image: np.ndarray, settings: DisplaySettings) -> np.ndarray:
    """Render one HU slice to uint8 without mutating the source array."""
    values = np.asarray(image, dtype=np.float32).copy()
    values = np.nan_to_num(
        values,
        nan=settings.clip_min,
        neginf=settings.clip_min,
        posinf=settings.clip_max,
    )
    if settings.clip_enabled:
        np.clip(values, settings.clip_min, settings.clip_max, out=values)
    width = max(float(settings.window_width), 1.0)
    low = float(settings.window_center) - width / 2.0
    normalized = np.clip((values - low) / width, 0.0, 1.0)

    if settings.transform == "Gamma":
        normalized = exposure.adjust_gamma(normalized, gamma=settings.gamma)
    elif settings.transform == "Sigmoid":
        normalized = exposure.adjust_sigmoid(
            normalized,
            cutoff=settings.sigmoid_cutoff,
            gain=settings.sigmoid_gain,
        )
    elif settings.transform != "None":
        raise ValueError(f"Unsupported display transform: {settings.transform}")

    if settings.clahe_enabled:
        normalized = exposure.equalize_adapthist(
            normalized,
            clip_limit=settings.clahe_clip_limit,
            nbins=256,
        )
    return np.round(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
