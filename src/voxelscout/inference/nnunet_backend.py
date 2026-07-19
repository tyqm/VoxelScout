"""Subprocess-based nnU-Net v2 inference without importing it into the GUI."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.desktop_data import ProgressCallback, SegmentationVolume
from voxelscout.inference.backend import InferenceUnavailableError
from voxelscout.inference.labels import training_to_verse_labels


@dataclass(frozen=True)
class NnUNetConfig:
    results_dir: Path
    dataset: str = "001"
    configuration: str = "3d_lowres"
    folds: tuple[str, ...] = ("0",)
    plans: str = "nnUNetResEncUNetMPlans"
    trainer: str = "nnUNetTrainer"
    checkpoint: str = "checkpoint_final.pth"
    command: str = "nnUNetv2_predict"

    @classmethod
    def from_environment(cls) -> "NnUNetConfig":
        results = os.environ.get("VOXELSCOUT_NNUNET_RESULTS") or os.environ.get(
            "nnUNet_results"
        )
        if not results:
            raise InferenceUnavailableError(
                "Automatic segmentation is not configured. Set "
                "VOXELSCOUT_NNUNET_RESULTS to an nnU-Net results directory."
            )
        folds = tuple(os.environ.get("VOXELSCOUT_NNUNET_FOLDS", "0").split())
        return cls(
            results_dir=Path(results),
            dataset=os.environ.get("VOXELSCOUT_NNUNET_DATASET", "001"),
            configuration=os.environ.get(
                "VOXELSCOUT_NNUNET_CONFIGURATION", "3d_lowres"
            ),
            folds=folds or ("0",),
            plans=os.environ.get(
                "VOXELSCOUT_NNUNET_PLANS", "nnUNetResEncUNetMPlans"
            ),
            trainer=os.environ.get("VOXELSCOUT_NNUNET_TRAINER", "nnUNetTrainer"),
            checkpoint=os.environ.get(
                "VOXELSCOUT_NNUNET_CHECKPOINT", "checkpoint_final.pth"
            ),
            command=os.environ.get("VOXELSCOUT_NNUNET_COMMAND", "nnUNetv2_predict"),
        )

    @property
    def identity(self) -> str:
        model = self.model_directory()
        tracked_files = [model / "plans.json"] + [
            model / f"fold_{fold}" / self.checkpoint for fold in self.folds
        ]
        model_state = tuple(
            (
                str(path.resolve()),
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            if path.is_file()
            else (str(path.resolve()), "missing")
            for path in tracked_files
        )
        fields = (
            self.results_dir.resolve(),
            self.dataset,
            self.configuration,
            self.folds,
            self.plans,
            self.trainer,
            self.checkpoint,
            self.command,
            model_state,
        )
        return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()

    def model_directory(self) -> Path:
        dataset_name = (
            self.dataset
            if self.dataset.startswith("Dataset")
            else f"Dataset{int(self.dataset):03d}_VerSe"
        )
        return (
            self.results_dir
            / dataset_name
            / f"{self.trainer}__{self.plans}__{self.configuration}"
        )

    def validate(self) -> None:
        if shutil.which(self.command) is None:
            raise InferenceUnavailableError(
                f"Automatic segmentation command was not found: {self.command}. "
                "Install nnU-Net v2 in the inference environment."
            )
        model = self.model_directory()
        if not model.is_dir():
            raise InferenceUnavailableError(
                f"Configured nnU-Net model directory does not exist: {model}"
            )
        missing = [
            str(model / f"fold_{fold}" / self.checkpoint)
            for fold in self.folds
            if not (model / f"fold_{fold}" / self.checkpoint).is_file()
        ]
        if missing:
            raise InferenceUnavailableError(
                "Configured nnU-Net checkpoint was not found: " + ", ".join(missing)
            )


class NnUNetBackend:
    def __init__(self, config: NnUNetConfig) -> None:
        self.config = config

    @classmethod
    def from_environment(cls) -> "NnUNetBackend":
        return cls(NnUNetConfig.from_environment())

    @property
    def name(self) -> str:
        return "nnU-Net v2"

    @property
    def cache_key(self) -> str:
        return self.config.identity

    def predict(
        self,
        ct_path: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> SegmentationVolume:
        report = progress or (lambda _value, _message: None)
        self.config.validate()
        ct_path = Path(ct_path).resolve()
        report(18, "Preparing inference")
        with tempfile.TemporaryDirectory(prefix="voxelscout-nnunet-") as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            input_path = input_dir / "case_0000.nii.gz"
            shutil.copyfile(ct_path, input_path)
            command = [
                self.config.command,
                "-d",
                self.config.dataset,
                "-c",
                self.config.configuration,
                "-f",
                *self.config.folds,
                "-p",
                self.config.plans,
                "-tr",
                self.config.trainer,
                "-chk",
                self.config.checkpoint,
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
            ]
            environment = os.environ.copy()
            environment["nnUNet_results"] = str(self.config.results_dir.resolve())
            report(25, "Running vertebra segmentation")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"nnU-Net inference failed (exit {completed.returncode}): "
                    f"{detail or 'no diagnostic output'}"
                )
            prediction_path = output_dir / "case.nii.gz"
            if not prediction_path.is_file():
                raise RuntimeError(
                    "nnU-Net completed but did not create the expected prediction: "
                    f"{prediction_path}"
                )
            prediction = nib.load(str(prediction_path))
            labels = training_to_verse_labels(np.asanyarray(prediction.dataobj))
            return SegmentationVolume(
                labels=labels,
                affine=np.asarray(prediction.affine, dtype=float),
                source=f"{self.name}:{self.cache_key}",
            )
