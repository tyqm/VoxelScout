"""Build a patient-level CSV inventory for a restructured VerSe dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.inspect_case import load_canonical, validate_pair


@dataclass(frozen=True)
class CaseRecord:
    subject: str
    image: str
    label: str
    shape_x: int
    shape_y: int
    shape_z: int
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float
    fov_x_mm: float
    fov_y_mm: float
    fov_z_mm: float
    orientation: str
    label_count: int
    labels: str
    foreground_voxels: int
    aligned: bool


def find_pairs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Find one CT and vertebra mask for every subject."""
    raw_root = dataset_root / "rawdata"
    derivatives_root = dataset_root / "derivatives"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing rawdata directory: {raw_root}")
    if not derivatives_root.is_dir():
        raise FileNotFoundError(f"Missing derivatives directory: {derivatives_root}")

    pairs: list[tuple[str, Path, Path]] = []
    for image_path in sorted(raw_root.glob("sub-*/*_ct.nii.gz")):
        subject = image_path.parent.name
        candidates = sorted(
            (derivatives_root / subject).glob("*_seg-vert_msk.nii.gz")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one vertebra mask for {subject}, found {len(candidates)}"
            )
        pairs.append((subject, image_path, candidates[0]))

    if not pairs:
        raise FileNotFoundError(f"No *_ct.nii.gz files found under {raw_root}")
    return pairs


def inspect_pair(
    subject: str,
    image_path: Path,
    label_path: Path,
    *,
    relative_to: Path,
) -> CaseRecord:
    """Collect geometry and label statistics for one aligned case."""
    ct = load_canonical(image_path)
    label = load_canonical(label_path)
    validate_pair(ct, label)

    shape = tuple(int(value) for value in ct.shape)
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(ct.affine))
    fov = tuple(shape[i] * spacing[i] for i in range(3))
    orientation = "".join(nib.aff2axcodes(ct.affine))

    label_data = np.asanyarray(label.dataobj)
    labels = tuple(int(value) for value in np.unique(label_data))
    foreground = int(np.count_nonzero(label_data))

    return CaseRecord(
        subject=subject,
        image=str(image_path.relative_to(relative_to)),
        label=str(label_path.relative_to(relative_to)),
        shape_x=shape[0],
        shape_y=shape[1],
        shape_z=shape[2],
        spacing_x_mm=spacing[0],
        spacing_y_mm=spacing[1],
        spacing_z_mm=spacing[2],
        fov_x_mm=fov[0],
        fov_y_mm=fov[1],
        fov_z_mm=fov[2],
        orientation=orientation,
        label_count=len(labels) - int(0 in labels),
        labels=";".join(str(value) for value in labels),
        foreground_voxels=foreground,
        aligned=True,
    )


def build_inventory(dataset_root: Path, output: Path) -> list[CaseRecord]:
    """Scan all cases and save a reproducible CSV manifest."""
    dataset_root = dataset_root.resolve()
    pairs = find_pairs(dataset_root)
    records = [
        inspect_pair(subject, image, label, relative_to=dataset_root)
        for subject, image, label in pairs
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def print_summary(records: list[CaseRecord], output: Path) -> None:
    spacings = np.asarray(
        [
            (record.spacing_x_mm, record.spacing_y_mm, record.spacing_z_mm)
            for record in records
        ]
    )
    shapes = np.asarray(
        [(record.shape_x, record.shape_y, record.shape_z) for record in records]
    )
    print(f"Cases:          {len(records)}")
    print(f"Shape min:      {tuple(shapes.min(axis=0))}")
    print(f"Shape max:      {tuple(shapes.max(axis=0))}")
    print(
        "Spacing min:    "
        f"{tuple(round(value, 3) for value in spacings.min(axis=0))} mm"
    )
    print(
        "Spacing median: "
        f"{tuple(round(value, 3) for value in np.median(spacings, axis=0))} mm"
    )
    print(
        "Spacing max:    "
        f"{tuple(round(value, 3) for value in spacings.max(axis=0))} mm"
    )
    print(f"Saved manifest: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CSV inventory for restructured VerSe data."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Directory containing rawdata/ and derivatives/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_inventory.csv"),
        help="Output CSV path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_inventory(args.dataset_root, args.output)
    print_summary(records, args.output)


if __name__ == "__main__":
    main()
