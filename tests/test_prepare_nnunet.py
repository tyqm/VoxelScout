import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from voxelscout.prepare_nnunet import (
    DATASET_NAME,
    PreparationError,
    SourceCase,
    case_identifier,
    output_filenames,
    prepare_dataset,
    remap_verse_labels,
    validate_case,
)


def write_case(
    root: Path,
    subject: str,
    labels: tuple[int, ...] = (1, 20, 28),
    *,
    image_affine: np.ndarray | None = None,
    label_affine: np.ndarray | None = None,
    label_shape: tuple[int, int, int] | None = None,
) -> tuple[Path, Path]:
    shape = (9, 8, 7)
    affine = np.diag([0.8, 1.2, 1.6, 1.0])
    image_affine = affine if image_affine is None else image_affine
    label_affine = image_affine if label_affine is None else label_affine
    label_shape = shape if label_shape is None else label_shape
    image = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    mask = np.zeros(label_shape, dtype=np.uint8)
    for index, value in enumerate(labels):
        mask[index + 1, 2:5, 2:5] = value

    image_dir = root / "rawdata" / subject
    label_dir = root / "derivatives" / subject
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / f"{subject}_ct.nii.gz"
    label_path = label_dir / f"{subject}_seg-vert_msk.nii.gz"
    nib.save(nib.Nifti1Image(image, image_affine), image_path)
    nib.save(nib.Nifti1Image(mask, label_affine), label_path)
    return image_path, label_path


def source_case(root: Path, subject: str = "sub-test001") -> SourceCase:
    return SourceCase(
        subject=subject,
        case_id=case_identifier(subject),
        split="training",
        image_path=root / "rawdata" / subject / f"{subject}_ct.nii.gz",
        label_path=(
            root
            / "derivatives"
            / subject
            / f"{subject}_seg-vert_msk.nii.gz"
        ),
    )


def test_filename_generation() -> None:
    assert case_identifier("sub-gl001") == "VerSe20_gl001"
    assert output_filenames("sub-gl001") == (
        "VerSe20_gl001_0000.nii.gz",
        "VerSe20_gl001.nii.gz",
    )


@pytest.mark.parametrize("problem", ["shape", "affine"])
def test_ct_mask_geometry_validation(tmp_path: Path, problem: str) -> None:
    root = tmp_path / "training"
    kwargs = {}
    if problem == "shape":
        kwargs["label_shape"] = (8, 8, 7)
    else:
        shifted = np.diag([0.8, 1.2, 1.6, 1.0])
        shifted[0, 3] = 2.0
        kwargs["label_affine"] = shifted
    write_case(root, "sub-test001", **kwargs)

    with pytest.raises(ValueError, match=f"{problem} mismatch"):
        validate_case(source_case(root))


def test_remaps_28_to_26_and_preserves_zero_through_25() -> None:
    source = np.asarray([*range(26), 28], dtype=np.uint8)
    converted = remap_verse_labels(source)

    assert np.array_equal(converted[:26], np.arange(26, dtype=np.uint8))
    assert converted[-1] == 26
    assert converted.dtype == np.uint8


@pytest.mark.parametrize(
    "values, expected",
    [([0, 26], "unsupported"), ([0, 27], "unsupported"), ([0, 2.5], "non-integer")],
)
def test_rejects_unsupported_or_non_integer_labels(values, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        remap_verse_labels(np.asarray(values))


def test_prevents_subject_leakage(tmp_path: Path) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    write_case(training, "sub-shared")
    write_case(validation, "sub-shared")

    with pytest.raises(PreparationError, match="both training and holdout"):
        prepare_dataset(
            training,
            validation,
            tmp_path / "nnUNet_raw",
            tmp_path / "holdout",
            dry_run=True,
        )


def test_dry_run_creates_no_medical_image_output(tmp_path: Path) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    write_case(training, "sub-train001")
    write_case(validation, "sub-hold001", (2, 28))
    nnunet_raw = tmp_path / "nnUNet_raw"
    holdout = tmp_path / "holdout"

    result = prepare_dataset(
        training, validation, nnunet_raw, holdout, dry_run=True
    )

    assert result.training_cases == 1
    assert result.holdout_cases == 1
    assert not nnunet_raw.exists()
    assert not holdout.exists()
    assert not list(tmp_path.rglob("*_0000.nii.gz"))


def test_writes_nnunet_dataset_holdout_and_metadata_idempotently(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    source_image, _ = write_case(training, "sub-train001", (1, 25, 28))
    write_case(validation, "sub-hold001", (7, 20))
    nnunet_raw = tmp_path / "nnUNet_raw"
    holdout = tmp_path / "holdout"

    first = prepare_dataset(training, validation, nnunet_raw, holdout)
    second = prepare_dataset(training, validation, nnunet_raw, holdout)
    dataset_dir = nnunet_raw / DATASET_NAME
    training_image = dataset_dir / "imagesTr" / "VerSe20_train001_0000.nii.gz"
    training_label = dataset_dir / "labelsTr" / "VerSe20_train001.nii.gz"
    holdout_image = holdout / "images" / "VerSe20_hold001_0000.nii.gz"
    holdout_label = holdout / "labels" / "VerSe20_hold001.nii.gz"

    assert first == second
    assert training_image.read_bytes() == source_image.read_bytes()
    assert set(np.unique(np.asanyarray(nib.load(str(training_label)).dataobj))) == {
        0,
        1,
        25,
        26,
    }
    assert holdout_image.is_file()
    assert holdout_label.is_file()

    dataset_json = json.loads((dataset_dir / "dataset.json").read_text())
    mapping = json.loads((dataset_dir / "label_mapping.json").read_text())
    summary = json.loads((dataset_dir / "preparation_summary.json").read_text())
    assert dataset_json["channel_names"] == {"0": "CT"}
    assert dataset_json["numTraining"] == 1
    assert dataset_json["labels"]["T13"] == 26
    assert mapping["verse_to_training"]["28"] == 26
    assert mapping["training_to_verse"]["26"] == 28
    assert summary["case_counts"] == {
        "training": 1,
        "holdout": 1,
        "total": 2,
        "valid": 2,
    }
    assert summary["validation_errors"] == []

    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["split"] for row in rows] == ["training", "holdout"]
    assert rows[0]["original_labels"] == "0;1;25;28"
    assert rows[0]["converted_labels"] == "0;1;25;26"


def test_refuses_incompatible_existing_output(tmp_path: Path) -> None:
    training = tmp_path / "training"
    write_case(training, "sub-train001")
    nnunet_raw = tmp_path / "nnUNet_raw"
    target = (
        nnunet_raw
        / DATASET_NAME
        / "imagesTr"
        / "VerSe20_train001_0000.nii.gz"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the source CT")

    with pytest.raises(FileExistsError, match="incompatible CT"):
        prepare_dataset(
            training,
            None,
            nnunet_raw,
            tmp_path / "holdout",
        )


def test_real_validation_failure_writes_error_summary(tmp_path: Path) -> None:
    training = tmp_path / "training"
    write_case(training, "sub-invalid", (27,))
    nnunet_raw = tmp_path / "nnUNet_raw"

    with pytest.raises(PreparationError, match="unsupported VerSe labels"):
        prepare_dataset(
            training,
            None,
            nnunet_raw,
            tmp_path / "holdout",
        )

    summary_path = nnunet_raw / DATASET_NAME / "preparation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case_counts"]["training"] == 1
    assert summary["case_counts"]["valid"] == 0
    assert "unsupported VerSe labels" in summary["validation_errors"][0]
    assert not (nnunet_raw / DATASET_NAME / "imagesTr").exists()
