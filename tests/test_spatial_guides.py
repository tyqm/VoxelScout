from __future__ import annotations

import pytest

from voxelscout.spatial_guides import (
    axial_edge_labels,
    format_scale_length,
    nice_scale_length,
)


def test_axial_edges_follow_actual_display_orientation() -> None:
    assert axial_edge_labels(("R", "A", "S")) == {
        "left": "L",
        "right": "R",
        "top": "A",
        "bottom": "P",
    }
    assert axial_edge_labels(("L", "P", "S")) == {
        "left": "R",
        "right": "L",
        "top": "P",
        "bottom": "A",
    }


def test_axial_edges_reject_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="Unsupported orientation"):
        axial_edge_labels(("X", "A", "S"))


def test_scale_uses_readable_physical_length_that_fits() -> None:
    length, pixels = nice_scale_length(0.25, 120)
    assert length == 20
    assert pixels == 80
    assert format_scale_length(length) == "20 mm"


def test_scale_formats_centimetres_without_long_decimals() -> None:
    assert format_scale_length(100) == "10 cm"
