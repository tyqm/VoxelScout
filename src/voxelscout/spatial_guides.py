"""Small geometry helpers for human-readable 2D/3D spatial overlays."""

from __future__ import annotations

from collections.abc import Sequence


_OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def axial_edge_labels(orientation: Sequence[str]) -> dict[str, str]:
    """Return edge labels for ``flipud(slice.T)`` from actual volume axes."""
    if len(orientation) < 2 or any(code not in _OPPOSITE for code in orientation[:2]):
        raise ValueError(f"Unsupported orientation: {tuple(orientation)}")
    horizontal, vertical = orientation[:2]
    return {
        "left": _OPPOSITE[horizontal],
        "right": horizontal,
        "top": vertical,
        "bottom": _OPPOSITE[vertical],
    }


def nice_scale_length(
    mm_per_pixel: float,
    available_pixels: float,
    *,
    target_pixels: float = 90.0,
) -> tuple[float, float]:
    """Choose an easy-to-read physical length and its on-screen pixel length."""
    if mm_per_pixel <= 0 or available_pixels <= 0:
        return 0.0, 0.0
    choices = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
    fitting = [length for length in choices if length / mm_per_pixel <= available_pixels]
    candidates = fitting or [choices[0]]
    length = min(candidates, key=lambda value: abs(value / mm_per_pixel - target_pixels))
    return length, length / mm_per_pixel


def format_scale_length(length_mm: float) -> str:
    if length_mm >= 100 and length_mm % 10 == 0:
        return f"{length_mm / 10:g} cm"
    return f"{length_mm:g} mm"
