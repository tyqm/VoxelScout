"""Prepare restructured VerSe 2020 data for nnU-Net v2 without preprocessing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import nibabel as nib
import numpy as np

from voxelscout.inference.labels import (
    SUPPORTED_VERSE_LABELS,
    TRAINING_TO_VERSE,
    VERSE_TO_TRAINING,
    verse_to_training_labels,
)


DATASET_NAME = "Dataset501_VerSe20"
SUPPORTED_SOURCE_LABELS = SUPPORTED_VERSE_LABELS
AFFINE_ATOL = 1e-3
AFFINE_RTOL = 1e-5


def _label_names() -> dict[int, str]:
    names = {0: "background"}
    names.update({value: f"C{value}" for value in range(1, 8)})
    names.update({value: f"T{value - 7}" for value in range(8, 20)})
    names.update({value: f"L{value - 19}" for value in range(20, 26)})
    names[26] = "T13"
    return names


TRAINING_LABEL_NAMES = _label_names()


class PreparationError(ValueError):
    """Raised after collecting one or more dataset validation errors."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(str(error) for error in errors)
        super().__init__("Dataset preparation failed:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class SourceCase:
    subject: str
    case_id: str
    split: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class ValidatedCase:
    source: SourceCase
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    orientation: str
    original_labels: tuple[int, ...]
    converted_labels: tuple[int, ...]


@dataclass(frozen=True)
class PreparationResult:
    dataset_dir: Path
    holdout_dir: Path
    training_cases: int
    holdout_cases: int
    dry_run: bool


def case_identifier(subject: str) -> str:
    """Return a deterministic nnU-Net-safe identifier for a VerSe subject."""
    token = subject[4:] if subject.casefold().startswith("sub-") else subject
    token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_")
    if not token:
        raise ValueError(f"Cannot derive a case identifier from subject {subject!r}")
    return f"VerSe20_{token}"


def output_filenames(subject: str) -> tuple[str, str]:
    case_id = case_identifier(subject)
    return f"{case_id}_0000.nii.gz", f"{case_id}.nii.gz"


def discover_cases(root: Path, split: str) -> tuple[list[SourceCase], list[str]]:
    """Discover official restructured CT/mask pairs and report missing files."""
    root = Path(root).resolve()
    raw_root = root / "rawdata"
    derivatives_root = root / "derivatives"
    errors: list[str] = []
    if not raw_root.is_dir():
        errors.append(f"{split}: missing rawdata directory: {raw_root}")
    if not derivatives_root.is_dir():
        errors.append(f"{split}: missing derivatives directory: {derivatives_root}")
    if errors:
        return [], errors

    subjects = sorted(
        {
            path.name
            for parent in (raw_root, derivatives_root)
            for path in parent.glob("sub-*")
            if path.is_dir()
        },
        key=str.casefold,
    )
    cases: list[SourceCase] = []
    for subject in subjects:
        images = sorted((raw_root / subject).glob("*_ct.nii.gz"))
        labels = sorted(
            (derivatives_root / subject).glob("*_seg-vert_msk.nii.gz")
        )
        if len(images) != 1:
            errors.append(
                f"{split}/{subject}: expected one CT, found {len(images)}"
            )
        if len(labels) != 1:
            errors.append(
                f"{split}/{subject}: expected one segmentation, found {len(labels)}"
            )
        if len(images) == 1 and len(labels) == 1:
            try:
                identifier = case_identifier(subject)
            except ValueError as error:
                errors.append(f"{split}/{subject}: {error}")
                continue
            cases.append(
                SourceCase(
                    subject=subject,
                    case_id=identifier,
                    split=split,
                    image_path=images[0].resolve(),
                    label_path=labels[0].resolve(),
                )
            )
    if not subjects:
        errors.append(f"{split}: no sub-* directories found under {root}")
    return cases, errors


def _source_label_values(data: np.ndarray) -> tuple[int, ...]:
    """Validate and return the small set of unique source label values."""
    values = np.asanyarray(data)
    unique_values = np.unique(values)
    if not np.all(np.isfinite(unique_values)):
        raise ValueError("mask contains non-finite values")
    rounded = np.rint(unique_values)
    if not np.allclose(unique_values, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("mask contains non-integer label values")
    labels = tuple(sorted(int(value) for value in rounded))
    unsupported = sorted(set(labels) - SUPPORTED_SOURCE_LABELS)
    if unsupported:
        raise ValueError(f"unsupported VerSe labels: {unsupported}")
    return labels


def remap_verse_labels(data: np.ndarray) -> np.ndarray:
    """Validate VerSe labels and remap source label 28 to consecutive label 26."""
    return verse_to_training_labels(data)


def validate_case(case: SourceCase) -> ValidatedCase:
    """Validate source geometry and labels without changing image orientation."""
    if not case.image_path.is_file():
        raise ValueError(f"CT does not exist: {case.image_path}")
    if not case.label_path.is_file():
        raise ValueError(f"segmentation does not exist: {case.label_path}")
    image = nib.load(str(case.image_path))
    label = nib.load(str(case.label_path))
    if len(image.shape) != 3 or len(label.shape) != 3:
        raise ValueError(
            f"expected 3D volumes, got CT {image.shape} and mask {label.shape}"
        )
    if image.shape != label.shape:
        raise ValueError(f"shape mismatch: CT {image.shape}, mask {label.shape}")
    if not np.allclose(
        image.affine,
        label.affine,
        atol=AFFINE_ATOL,
        rtol=AFFINE_RTOL,
    ):
        raise ValueError("affine mismatch: CT and mask are not spatially aligned")

    original_data = np.asanyarray(label.dataobj)
    original_labels = _source_label_values(original_data)
    converted_labels = tuple(
        sorted({VERSE_TO_TRAINING[value] for value in original_labels})
    )
    return ValidatedCase(
        source=case,
        shape=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(image.affine)),
        orientation="".join(nib.aff2axcodes(image.affine)),
        original_labels=original_labels,
        converted_labels=converted_labels,
    )


def _validate_collection(
    training: Sequence[SourceCase],
    holdout: Sequence[SourceCase],
    progress: Callable[[int, int, SourceCase], None] | None = None,
) -> tuple[list[ValidatedCase], list[str]]:
    errors: list[str] = []
    training_subjects = {case.subject.casefold() for case in training}
    holdout_subjects = {case.subject.casefold() for case in holdout}
    overlap = sorted(training_subjects & holdout_subjects)
    if overlap:
        errors.append(
            "subjects appear in both training and holdout: " + ", ".join(overlap)
        )

    identifiers: dict[str, SourceCase] = {}
    for case in (*training, *holdout):
        key = case.case_id.casefold()
        previous = identifiers.get(key)
        if previous is not None:
            errors.append(
                f"duplicate case identifier {case.case_id}: "
                f"{previous.subject} and {case.subject}"
            )
        else:
            identifiers[key] = case

    validated: list[ValidatedCase] = []
    all_cases = (*training, *holdout)
    for index, case in enumerate(all_cases, start=1):
        if progress is not None:
            progress(index, len(all_cases), case)
        try:
            validated.append(validate_case(case))
        except Exception as error:
            errors.append(f"{case.split}/{case.subject}: {error}")
    return validated, errors


def _dataset_json(training_count: int) -> dict[str, object]:
    return {
        "channel_names": {"0": "CT"},
        "labels": {
            name: value for value, name in TRAINING_LABEL_NAMES.items()
        },
        "numTraining": training_count,
        "file_ending": ".nii.gz",
    }


def _mapping_json() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "VerSe 2020",
        "verse_to_training": {
            str(key): value for key, value in sorted(VERSE_TO_TRAINING.items())
        },
        "training_to_verse": {
            str(key): value for key, value in sorted(TRAINING_TO_VERSE.items())
        },
        "training_labels": {
            str(key): value for key, value in TRAINING_LABEL_NAMES.items()
        },
    }


def _output_paths(
    case: SourceCase, dataset_dir: Path, holdout_dir: Path
) -> tuple[Path, Path]:
    image_name, label_name = output_filenames(case.subject)
    if case.split == "training":
        return dataset_dir / "imagesTr" / image_name, dataset_dir / "labelsTr" / label_name
    return holdout_dir / "images" / image_name, holdout_dir / "labels" / label_name


def _manifest_rows(
    cases: Iterable[ValidatedCase], dataset_dir: Path, holdout_dir: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in sorted(
        cases,
        key=lambda item: (item.source.split != "training", item.source.subject.casefold()),
    ):
        output_image, output_label = _output_paths(
            case.source, dataset_dir, holdout_dir
        )
        rows.append(
            {
                "subject": case.source.subject,
                "case_id": case.source.case_id,
                "split": case.source.split,
                "source_image": str(case.source.image_path),
                "source_label": str(case.source.label_path),
                "output_image": str(output_image),
                "output_label": str(output_label),
                "shape": "x".join(str(value) for value in case.shape),
                "spacing_mm": ";".join(f"{value:.6g}" for value in case.spacing),
                "orientation": case.orientation,
                "original_labels": ";".join(
                    str(value) for value in case.original_labels
                ),
                "converted_labels": ";".join(
                    str(value) for value in case.converted_labels
                ),
            }
        )
    return rows


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _csv_text(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("manifest requires at least one case")
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_text_compatible(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return "reused"
        raise FileExistsError(f"Refusing to overwrite incompatible file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return "written"


def _write_summary_report(path: Path, value: object) -> str:
    """Write the current validation report, explicitly updating stale reports."""
    encoded = _json_text(value).encode("utf-8")
    if path.exists() and path.is_file() and path.read_bytes() == encoded:
        return "reused"
    action = "updated" if path.exists() else "written"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return action


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_compatible(source: Path, target: Path) -> str:
    if target.exists():
        if target.is_file() and _sha256(source) == _sha256(target):
            return "reused"
        raise FileExistsError(f"Refusing to overwrite incompatible CT: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return "written"


def _converted_label_image(case: ValidatedCase) -> tuple[nib.Nifti1Image, np.ndarray]:
    source = nib.load(str(case.source.label_path))
    converted = remap_verse_labels(np.asanyarray(source.dataobj))
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(converted, source.affine, header=header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))
    return image, converted


def _write_label_compatible(case: ValidatedCase, target: Path) -> str:
    image, converted = _converted_label_image(case)
    if target.exists():
        existing = nib.load(str(target))
        compatible = (
            existing.shape == image.shape
            and np.allclose(
                existing.affine,
                image.affine,
                atol=AFFINE_ATOL,
                rtol=AFFINE_RTOL,
            )
            and np.dtype(existing.get_data_dtype()) == np.dtype(np.uint8)
            and np.array_equal(np.asanyarray(existing.dataobj), converted)
        )
        if compatible:
            return "reused"
        raise FileExistsError(f"Refusing to overwrite incompatible label: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=".nii.gz", dir=target.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _print_plan(cases: Sequence[ValidatedCase], dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "PREPARE"
    training = sum(case.source.split == "training" for case in cases)
    holdout = len(cases) - training
    print(f"[{mode}] validated {training} training and {holdout} holdout cases")
    for case in cases:
        labels = ",".join(str(value) for value in case.converted_labels)
        print(
            f"[{mode}] {case.source.split:8} {case.source.subject} -> "
            f"{case.source.case_id} | labels={labels}"
        )


def prepare_dataset(
    training_root: Path,
    validation_root: Path | None,
    nnunet_raw: Path,
    holdout_output: Path,
    *,
    dry_run: bool = False,
) -> PreparationResult:
    """Validate and prepare VerSe 2020 training and official validation data."""
    dataset_dir = Path(nnunet_raw).resolve() / DATASET_NAME
    holdout_dir = Path(holdout_output).resolve()
    training, discovery_errors = discover_cases(training_root, "training")
    holdout: list[SourceCase] = []
    if validation_root is not None:
        holdout, validation_discovery_errors = discover_cases(
            validation_root, "holdout"
        )
        discovery_errors.extend(validation_discovery_errors)

    validated, validation_errors = _validate_collection(
        training,
        holdout,
        progress=lambda index, total, case: print(
            f"[VALIDATE] {index}/{total} {case.split}/{case.subject}", flush=True
        ),
    )
    errors = [*discovery_errors, *validation_errors]
    if dataset_dir == holdout_dir:
        errors.append("nnU-Net dataset and holdout directories must differ")
    if errors:
        print(f"[ERROR] validation failed with {len(errors)} error(s)")
        for error in errors:
            print(f"[ERROR] {error}")
        if not dry_run:
            failure_summary = {
                "schema_version": 1,
                "dataset": DATASET_NAME,
                "case_counts": {
                    "training": len(training),
                    "holdout": len(holdout),
                    "total": len(training) + len(holdout),
                    "valid": len(validated),
                },
                "validation_errors": errors,
            }
            summary_path = dataset_dir / "preparation_summary.json"
            action = _write_summary_report(summary_path, failure_summary)
            print(f"[METADATA] {summary_path.name}: {action}")
        raise PreparationError(errors)

    _print_plan(validated, dry_run)

    training_count = sum(case.source.split == "training" for case in validated)
    holdout_count = len(validated) - training_count
    result = PreparationResult(
        dataset_dir=dataset_dir,
        holdout_dir=holdout_dir,
        training_cases=training_count,
        holdout_cases=holdout_count,
        dry_run=dry_run,
    )
    if dry_run:
        print("[DRY RUN] no files were written")
        return result

    for case in validated:
        output_image, output_label = _output_paths(
            case.source, dataset_dir, holdout_dir
        )
        image_action = _copy_compatible(case.source.image_path, output_image)
        label_action = _write_label_compatible(case, output_label)
        print(
            f"[{case.source.split.upper()}] {case.source.case_id}: "
            f"image={image_action}, label={label_action}"
        )

    rows = _manifest_rows(validated, dataset_dir, holdout_dir)
    summary = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "case_counts": {
            "training": training_count,
            "holdout": holdout_count,
            "total": len(validated),
            "valid": len(validated),
        },
        "validation_errors": [],
    }
    metadata = {
        dataset_dir / "dataset.json": _json_text(_dataset_json(training_count)),
        dataset_dir / "label_mapping.json": _json_text(_mapping_json()),
        dataset_dir / "manifest.csv": _csv_text(rows),
    }
    for path, text in metadata.items():
        action = _write_text_compatible(path, text)
        print(f"[METADATA] {path.name}: {action}")
    summary_path = dataset_dir / "preparation_summary.json"
    summary_action = _write_summary_report(summary_path, summary)
    print(f"[METADATA] {summary_path.name}: {summary_action}")
    print(f"[DONE] nnU-Net dataset: {dataset_dir}")
    print(f"[DONE] holdout: {holdout_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare restructured VerSe 2020 data for nnU-Net v2."
    )
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--nnunet-raw", required=True, type=Path)
    parser.add_argument("--holdout-output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        prepare_dataset(
            args.training_root,
            args.validation_root,
            args.nnunet_raw,
            args.holdout_output,
            dry_run=args.dry_run,
        )
    except (PreparationError, FileExistsError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
