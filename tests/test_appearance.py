from __future__ import annotations

from voxelscout.appearance import (
    AppearanceMode,
    NATURAL_BONE,
    UNKNOWN_COLOUR,
    colour_for_label,
    label_colour,
    region_colour,
)


def test_natural_mode_uses_one_bone_colour() -> None:
    assert colour_for_label(1, AppearanceMode.NATURAL) == NATURAL_BONE
    assert colour_for_label(24, AppearanceMode.NATURAL) == NATURAL_BONE


def test_regions_are_stable_and_distinct() -> None:
    cervical = colour_for_label(1, AppearanceMode.REGIONS)
    thoracic = colour_for_label(8, AppearanceMode.REGIONS)
    lumbar = colour_for_label(20, AppearanceMode.REGIONS)
    assert len({cervical, thoracic, lumbar}) == 3
    assert region_colour("Sacral spine") != UNKNOWN_COLOUR
    assert region_colour("Unexpected") == UNKNOWN_COLOUR


def test_label_colours_are_deterministic_and_adjacent_colours_differ() -> None:
    first_pass = [label_colour(label) for label in range(1, 41)]
    second_pass = [label_colour(label) for label in range(1, 41)]
    assert first_pass == second_pass
    assert all(left != right for left, right in zip(first_pass, first_pass[1:]))


def test_known_labels_have_unique_palette_colours() -> None:
    colours = [colour_for_label(label, AppearanceMode.LABELS) for label in range(1, 29)]
    assert len(colours) == len(set(colours))
