"""Inspect one VerSe NIfTI CT volume and its segmentation mask."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class VolumeInfo:
    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    minimum: float
    maximum: float
    labels: tuple[int, ...] = ()


def load_canonical(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file and reorient it to the closest canonical orientation."""
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    return nib.as_closest_canonical(nib.load(str(path)))


def describe(image: nib.Nifti1Image, *, is_label: bool = False) -> VolumeInfo:
    """Return geometry and value statistics without changing image geometry."""
    data = np.asanyarray(image.dataobj)
    labels = tuple(int(value) for value in np.unique(data)) if is_label else ()
    return VolumeInfo(
        shape=tuple(int(value) for value in data.shape),
        spacing_mm=tuple(
            float(value) for value in nib.affines.voxel_sizes(image.affine)
        ),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        labels=labels,
    )


def validate_pair(ct: nib.Nifti1Image, label: nib.Nifti1Image) -> None:
    """Raise a clear error when a CT and mask are not voxel-aligned."""
    if ct.shape != label.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape}, mask {label.shape}")
    if not np.allclose(ct.affine, label.affine, atol=1e-3):
        raise ValueError("Affine mismatch: CT and mask are not spatially aligned")


def foreground_centre(mask: np.ndarray) -> tuple[int, int, int]:
    """Choose the centre of the foreground bounding box, or volume centre."""
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return tuple(int(size // 2) for size in mask.shape)
    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0)
    return tuple(int(value) for value in ((lower + upper) // 2))


def window_ct(ct: np.ndarray, centre: float = 300.0, width: float = 1800.0) -> np.ndarray:
    """Apply a simple CT bone window and scale values to [0, 1]."""
    lower = centre - width / 2
    upper = centre + width / 2
    return np.clip((ct - lower) / (upper - lower), 0.0, 1.0)


def save_orthogonal_overlay(
    ct: np.ndarray,
    mask: np.ndarray,
    output: Path,
) -> None:
    """Save sagittal, coronal, and axial CT views with a binary mask overlay."""
    centre = foreground_centre(mask)
    views = (
        ("Sagittal", ct[centre[0], :, :], mask[centre[0], :, :]),
        ("Coronal", ct[:, centre[1], :], mask[:, centre[1], :]),
        ("Axial", ct[:, :, centre[2]], mask[:, :, centre[2]]),
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (title, ct_slice, mask_slice) in zip(axes, views, strict=True):
        ct_view = np.rot90(window_ct(ct_slice))
        mask_view = np.rot90(mask_slice > 0)
        axis.imshow(ct_view, cmap="gray", interpolation="nearest")
        axis.imshow(
            np.ma.masked_where(~mask_view, mask_view),
            cmap="autumn",
            alpha=0.45,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"VoxelScout orthogonal overlay | voxel centre={centre}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def inspect_case(image_path: Path, label_path: Path, output: Path) -> None:
    """Validate, summarize, and visualize one CT/mask pair."""
    ct_image = load_canonical(image_path)
    label_image = load_canonical(label_path)
    validate_pair(ct_image, label_image)

    ct_info = describe(ct_image)
    label_info = describe(label_image, is_label=True)

    ct = ct_image.get_fdata(dtype=np.float32)
    label = np.asanyarray(label_image.dataobj)

    print(f"CT path:       {image_path}")
    print(f"Mask path:     {label_path}")
    print(f"Shape:         {ct_info.shape}")
    print(f"Spacing (mm):  {tuple(round(v, 3) for v in ct_info.spacing_mm)}")
    print(f"CT range:      [{ct_info.minimum:.1f}, {ct_info.maximum:.1f}]")
    print(f"Mask range:    [{label_info.minimum:.0f}, {label_info.maximum:.0f}]")
    print(f"Mask labels:   {label_info.labels}")
    print(f"Foreground:    {int(np.count_nonzero(label)):,} voxels")

    save_orthogonal_overlay(ct, label, output)
    print(f"Saved overlay: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one aligned VerSe CT and segmentation mask."
    )
    parser.add_argument("--image", required=True, type=Path, help="CT .nii.gz path")
    parser.add_argument("--label", required=True, type=Path, help="Mask .nii.gz path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inspect_case.png"),
        help="Output PNG path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inspect_case(args.image, args.label, args.output)


if __name__ == "__main__":
    main()
