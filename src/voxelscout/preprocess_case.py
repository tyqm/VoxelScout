"""Preview coarse or fine MONAI preprocessing on one VerSe case."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from voxelscout.inspect_case import foreground_centre


def build_preprocess_transform(spacing: Sequence[float]) -> Compose:
    """Create deterministic image/mask transforms shared by training and inference."""
    pixdim = tuple(float(value) for value in spacing)
    if len(pixdim) != 3 or any(value <= 0 for value in pixdim):
        raise ValueError("spacing must contain three positive numbers")

    return Compose(
        [
            LoadImaged(keys=("image", "label"), image_only=True),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            Spacingd(
                keys=("image", "label"),
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000.0,
                a_max=2000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Lambdad(keys="label", func=lambda value: (value > 0).to(torch.uint8)),
        ]
    )


def save_overlay(image: np.ndarray, label: np.ndarray, output: Path) -> None:
    """Save normalized orthogonal views after resampling."""
    centre = foreground_centre(label)
    views = (
        ("Sagittal", image[centre[0], :, :], label[centre[0], :, :]),
        ("Coronal", image[:, centre[1], :], label[:, centre[1], :]),
        ("Axial", image[:, :, centre[2]], label[:, :, centre[2]]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (title, image_slice, label_slice) in zip(axes, views, strict=True):
        image_view = np.rot90(image_slice)
        label_view = np.rot90(label_slice > 0)
        axis.imshow(image_view, cmap="gray", vmin=0.0, vmax=1.0)
        axis.imshow(
            np.ma.masked_where(~label_view, label_view),
            cmap="autumn",
            alpha=0.45,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Preprocessed CT and binary mask | centre={centre}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def preprocess_case(
    image_path: Path,
    label_path: Path,
    spacing: Sequence[float],
    output: Path,
) -> dict:
    """Run deterministic preprocessing and save a visual QA image."""
    transform = build_preprocess_transform(spacing)
    result = transform({"image": str(image_path), "label": str(label_path)})
    image_tensor = result["image"]
    label_tensor = result["label"]

    image = image_tensor[0].cpu().numpy()
    label = label_tensor[0].cpu().numpy()
    actual_spacing = tuple(
        float(value)
        for value in nib.affines.voxel_sizes(image_tensor.affine.cpu().numpy())
    )

    save_overlay(image, label, output)
    print(f"Target spacing: {tuple(float(v) for v in spacing)} mm")
    print(f"Output spacing: {tuple(round(v, 3) for v in actual_spacing)} mm")
    print(f"Output shape:   {tuple(int(v) for v in image.shape)}")
    print(f"Image range:   [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask labels:   {tuple(int(v) for v in np.unique(label))}")
    print(f"Image memory:  {image.nbytes / 1024**2:.1f} MiB")
    print(f"Saved preview: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview deterministic MONAI preprocessing for one VerSe case."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Target voxel spacing in millimetres, e.g. 1 1 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/preprocess_preview.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess_case(args.image, args.label, args.spacing, args.output)


if __name__ == "__main__":
    main()
