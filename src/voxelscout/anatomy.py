"""Plain-language vertebra names for the patient-facing interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VertebraInfo:
    label: int
    code: str
    region: str
    plain_location: str


def _build_catalogue() -> dict[int, VertebraInfo]:
    catalogue: dict[int, VertebraInfo] = {}
    for label in range(1, 8):
        code = f"C{label}"
        catalogue[label] = VertebraInfo(
            label, code, "Cervical spine", f"Neck region — vertebra {code}"
        )
    for label in range(8, 20):
        number = label - 7
        code = f"T{number}"
        catalogue[label] = VertebraInfo(
            label, code, "Thoracic spine", f"Upper or middle back — vertebra {code}"
        )
    for label in range(20, 26):
        number = label - 19
        code = f"L{number}"
        catalogue[label] = VertebraInfo(
            label, code, "Lumbar spine", f"Lower back — vertebra {code}"
        )
    catalogue[28] = VertebraInfo(
        28,
        "T13",
        "Thoracic spine",
        "An additional thoracic vertebra sometimes present as an anatomical variation",
    )
    return catalogue


VERTEBRAE = _build_catalogue()


def vertebra_info(label: int) -> VertebraInfo:
    """Return a safe description even for a dataset-specific unknown label."""
    return VERTEBRAE.get(
        int(label),
        VertebraInfo(
            int(label),
            f"Label {int(label)}",
            "Unmapped region",
            "This label is not mapped to a standard vertebra name in VoxelScout.",
        ),
    )
