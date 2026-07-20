"""Centralised, deterministic colour presets for Lenx spine meshes."""

from __future__ import annotations

import colorsys
from enum import Enum

from voxelscout.anatomy import vertebra_info


class AppearanceMode(str, Enum):
    NATURAL = "Natural"
    REGIONS = "Regions"
    LABELS = "Labels"


NATURAL_BONE = "#d9d5c9"
UNKNOWN_COLOUR = "#9aa3ad"

REGION_COLOURS = {
    "Cervical spine": "#56b4e9",
    "Thoracic spine": "#9b7ed0",
    "Lumbar spine": "#e69f00",
    "Sacral spine": "#38a889",
    "Unmapped region": UNKNOWN_COLOUR,
}

# Deliberately ordered so adjacent labels do not receive neighbouring shades.
# The first colours draw from established colour-blind-conscious categorical sets.
LABEL_PALETTE = (
    "#0072b2", "#e69f00", "#cc79a7", "#009e73",
    "#d55e00", "#56b4e9", "#8b5cf6", "#f0c94a",
    "#00a6a6", "#ef6f6c", "#5b8c5a", "#b565c0",
    "#4c78a8", "#f58518", "#72b7b2", "#e45756",
    "#54a24b", "#b279a2", "#ff9da6", "#9d755d",
    "#2f4b7c", "#ffa600", "#665191", "#a05195",
    "#003f5c", "#f95d6a", "#2a9d8f", "#e9c46a",
    "#577590", "#f3722c", "#43aa8b", "#f8961e",
)


def region_colour(region: str) -> str:
    return REGION_COLOURS.get(region, UNKNOWN_COLOUR)


def label_colour(label: int) -> str:
    """Return one stable colour for a vertebra label across every case."""
    value = int(label)
    if 1 <= value <= len(LABEL_PALETTE):
        return LABEL_PALETTE[value - 1]
    # Deterministic golden-angle extension for dataset-specific labels.
    hue = ((value * 0.618033988749895) + 0.17) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.58, 0.62)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def colour_for_label(label: int, mode: AppearanceMode | str) -> str:
    selected = AppearanceMode(mode)
    if selected is AppearanceMode.NATURAL:
        return NATURAL_BONE
    if selected is AppearanceMode.REGIONS:
        return region_colour(vertebra_info(label).region)
    return label_colour(label)
