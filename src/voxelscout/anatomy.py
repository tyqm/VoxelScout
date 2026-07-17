"""Plain-language vertebra names for the VoxelScout desktop interface."""

from __future__ import annotations

from dataclasses import dataclass


ORDINALS = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth",
    6: "Sixth",
    7: "Seventh",
    8: "Eighth",
    9: "Ninth",
    10: "Tenth",
    11: "Eleventh",
    12: "Twelfth",
    13: "Thirteenth",
}


@dataclass(frozen=True)
class VertebraInfo:
    label: int
    code: str
    anatomical_name: str
    region: str
    region_plain: str
    plain_location: str
    explanation: str

    @property
    def tooltip(self) -> str:
        return f"{self.code}\n{self.anatomical_name}\n{self.region_plain}"


def _build_catalogue() -> dict[int, VertebraInfo]:
    catalogue: dict[int, VertebraInfo] = {}
    for label in range(1, 8):
        number = label
        code = f"C{number}"
        catalogue[label] = VertebraInfo(
            label=label,
            code=code,
            anatomical_name=f"{ORDINALS[number]} cervical vertebra",
            region="Cervical spine",
            region_plain="Neck region",
            plain_location=f"Neck region — vertebra {code}",
            explanation="One of the seven vertebrae that support and protect the neck.",
        )
    for label in range(8, 20):
        number = label - 7
        code = f"T{number}"
        catalogue[label] = VertebraInfo(
            label=label,
            code=code,
            anatomical_name=f"{ORDINALS[number]} thoracic vertebra",
            region="Thoracic spine",
            region_plain="Upper- and middle-back region",
            plain_location=f"Upper or middle back — vertebra {code}",
            explanation="Part of the thoracic spine, where the ribs connect to the back.",
        )
    for label in range(20, 26):
        number = label - 19
        code = f"L{number}"
        catalogue[label] = VertebraInfo(
            label=label,
            code=code,
            anatomical_name=f"{ORDINALS[number]} lumbar vertebra",
            region="Lumbar spine",
            region_plain="Lower-back region",
            plain_location=f"Lower back — vertebra {code}",
            explanation="One of the large vertebrae that carries load in the lower back.",
        )
    catalogue[28] = VertebraInfo(
        label=28,
        code="T13",
        anatomical_name="Thirteenth thoracic vertebra",
        region="Thoracic spine",
        region_plain="Upper- and middle-back region",
        plain_location="An additional thoracic vertebra",
        explanation=(
            "An anatomical variation with an additional vertebra in the thoracic spine."
        ),
    )
    return catalogue


VERTEBRAE = _build_catalogue()


def vertebra_info(label: int) -> VertebraInfo:
    """Return a safe description for a known or dataset-specific label."""
    value = int(label)
    return VERTEBRAE.get(
        value,
        VertebraInfo(
            label=value,
            code=f"Label {value}",
            anatomical_name="Unmapped vertebral label",
            region="Unmapped region",
            region_plain="Region not mapped",
            plain_location="This label is not mapped to a standard vertebra name.",
            explanation="VoxelScout does not have a plain-language mapping for this label.",
        ),
    )
