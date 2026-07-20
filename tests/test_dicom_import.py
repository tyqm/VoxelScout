from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from voxelscout.dicom_import import convert_dicom_series, discover_ct_series


def write_dicom_series(
    folder: Path,
    *,
    uid: str,
    slices: int = 20,
    description: str = "Spine CT",
    inconsistent_spacing_at: int | None = None,
) -> np.ndarray:
    folder.mkdir(parents=True, exist_ok=True)
    volume = np.empty((slices, 12, 16), dtype=np.int16)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index in range(slices):
        pixels = np.full((12, 16), -1000 + index * 20, dtype=np.int16)
        pixels[3:9, 4:12] = 200 + index
        volume[index] = pixels
        image = sitk.GetImageFromArray(pixels)
        row_spacing = 0.9 if index == inconsistent_spacing_at else 0.8
        image.SetSpacing((0.7, row_spacing, 1.0))
        spacing = f"{row_spacing}\\0.7"
        tags = {
            "0008|0060": "CT",
            "0008|0008": "ORIGINAL\\PRIMARY\\AXIAL",
            "0008|103e": description,
            "0020|000e": uid,
            "0020|000d": "1.2.826.0.1.3680043.2.1125.99",
            "0020|0013": str(index + 1),
            "0020|0032": f"0\\0\\{index * 1.5}",
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0028|0030": spacing,
            "0018|0050": "1.5",
            "0028|1052": "0",
            "0028|1053": "1",
        }
        for key, value in tags.items():
            image.SetMetaData(key, value)
        writer.SetFileName(str(folder / f"slice-{index:03d}.dcm"))
        writer.Execute(image)
    return volume


def test_discovers_ct_and_excludes_localizer(tmp_path: Path) -> None:
    write_dicom_series(
        tmp_path / "diagnostic",
        uid="1.2.826.0.1.3680043.2.1125.1",
        slices=20,
    )
    write_dicom_series(
        tmp_path / "scout",
        uid="1.2.826.0.1.3680043.2.1125.2",
        slices=3,
        description="CT Localizer",
    )

    series = discover_ct_series(tmp_path)

    assert len(series) == 1
    assert series[0].shape == (16, 12, 20)
    assert series[0].spacing == pytest.approx((0.7, 0.8, 1.5))


def test_conversion_preserves_hu_geometry_and_uses_cache(tmp_path: Path) -> None:
    expected = write_dicom_series(
        tmp_path / "dicom",
        uid="1.2.826.0.1.3680043.2.1125.3",
    )
    series = discover_ct_series(tmp_path)[0]
    updates: list[str] = []

    first = convert_dicom_series(
        series,
        cache_dir=tmp_path / "cache",
        progress=lambda _value, message: updates.append(message),
    )
    second = convert_dicom_series(series, cache_dir=tmp_path / "cache")
    converted = nib.load(str(first.nifti_path))

    assert first.cached is False
    assert second.cached is True
    assert first.nifti_path == second.nifti_path
    assert converted.shape == (16, 12, 20)
    assert converted.header.get_zooms()[:3] == pytest.approx((0.7, 0.8, 1.5))
    assert "".join(nib.aff2axcodes(converted.affine)) == "LPS"
    assert np.array_equal(np.asanyarray(converted.dataobj), expected.transpose(2, 1, 0))
    assert updates == ["Reading DICOM series", "Converting CT"]


def test_rejects_inconsistent_slice_geometry(tmp_path: Path) -> None:
    write_dicom_series(
        tmp_path / "dicom",
        uid="1.2.826.0.1.3680043.2.1125.4",
        inconsistent_spacing_at=8,
    )
    series = discover_ct_series(tmp_path)[0]

    with pytest.raises(ValueError, match="inconsistent in-plane spacing"):
        convert_dicom_series(series, cache_dir=tmp_path / "cache")
