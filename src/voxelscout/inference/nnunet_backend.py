"""Subprocess-based nnU-Net v2 inference without importing it into the GUI."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
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
    model_dir: Path | None = None
    folds: tuple[str, ...] = ("0",)
    checkpoint: str = "checkpoint_final.pth"
    command: str = "nnUNetv2_predict_from_modelfolder"
    device: str = "cpu"
    preprocessing_processes: int = 1
    export_processes: int = 1

    @classmethod
    def from_environment(cls) -> "NnUNetConfig":
        model = os.environ.get("VOXELSCOUT_NNUNET_MODEL_DIR")
        folds = tuple(os.environ.get("VOXELSCOUT_NNUNET_FOLDS", "0").split())
        return cls(
            model_dir=Path(model) if model else _discover_model_directory(),
            folds=folds or ("0",),
            checkpoint=os.environ.get(
                "VOXELSCOUT_NNUNET_CHECKPOINT", "checkpoint_final.pth"
            ),
            command=os.environ.get(
                "VOXELSCOUT_NNUNET_COMMAND"
            ) or _discover_predict_command(),
            device=os.environ.get("VOXELSCOUT_NNUNET_DEVICE", "cpu"),
            preprocessing_processes=int(
                os.environ.get("VOXELSCOUT_NNUNET_NPP", "1")
            ),
            export_processes=int(os.environ.get("VOXELSCOUT_NNUNET_NPS", "1")),
        )

    @property
    def identity(self) -> str:
        model = self.resolved_model_directory()
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
            model.resolve(),
            self.folds,
            self.checkpoint,
            self.command,
            self.device,
            self.preprocessing_processes,
            self.export_processes,
            model_state,
        )
        return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()

    def resolved_model_directory(self) -> Path:
        if self.model_dir is not None:
            return Path(self.model_dir).expanduser().resolve()
        raise InferenceUnavailableError(
            "Automatic segmentation model was not found. Set "
            "VOXELSCOUT_NNUNET_MODEL_DIR to the trained nnU-Net model folder."
        )

    def resolved_command(self) -> str:
        candidate = Path(self.command).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        discovered = shutil.which(self.command)
        if discovered:
            return discovered
        raise InferenceUnavailableError(
            f"Automatic segmentation command was not found: {self.command}. "
            "Set VOXELSCOUT_NNUNET_COMMAND to "
            "nnUNetv2_predict_from_modelfolder in the inference environment."
        )

    def validate(self) -> None:
        self.resolved_command()
        model = self.resolved_model_directory()
        if not model.is_dir():
            raise InferenceUnavailableError(
                f"Configured nnU-Net model directory does not exist: {model}"
            )
        metadata_missing = [
            str(model / name)
            for name in ("dataset.json", "plans.json")
            if not (model / name).is_file()
        ]
        if metadata_missing:
            raise InferenceUnavailableError(
                "Configured nnU-Net model metadata was not found: "
                + ", ".join(metadata_missing)
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
        report(18, "Real model inference · preparing")
        with tempfile.TemporaryDirectory(prefix="voxelscout-nnunet-") as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            input_path = input_dir / "case_0000.nii.gz"
            shutil.copyfile(ct_path, input_path)
            command = [
                self.config.resolved_command(),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-m",
                str(self.config.resolved_model_directory()),
                "-f",
                *self.config.folds,
                "-chk",
                self.config.checkpoint,
                "-npp",
                str(self.config.preprocessing_processes),
                "-nps",
                str(self.config.export_processes),
                "-device",
                self.config.device,
                "--disable_progress_bar",
            ]
            environment = os.environ.copy()
            report(25, "Real model inference · running nnU-Net")
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


def _discover_predict_command() -> str:
    name = (
        "nnUNetv2_predict_from_modelfolder.exe"
        if os.name == "nt"
        else "nnUNetv2_predict_from_modelfolder"
    )
    discovered = shutil.which(name)
    if discovered:
        return discovered
    candidates = [
        Path(sys.prefix).parent / "verse-pretrained" / "Scripts" / name,
        Path.home() / "miniforge3" / "envs" / "verse-pretrained" / "Scripts" / name,
        Path.home() / "miniconda3" / "envs" / "verse-pretrained" / "Scripts" / name,
    ]
    return str(next((path for path in candidates if path.is_file()), Path(name)))


def _discover_model_directory() -> Path | None:
    model_name = "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres"
    source_checkout = Path(__file__).resolve().parents[3]
    candidates = [
        source_checkout.parent
        / "VoxelScout-ML"
        / "downloads"
        / "verse-pretrained"
        / "nnUNet_results"
        / model_name,
        Path.cwd().parent
        / "VoxelScout-ML"
        / "downloads"
        / "verse-pretrained"
        / "nnUNet_results"
        / model_name,
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), None)
