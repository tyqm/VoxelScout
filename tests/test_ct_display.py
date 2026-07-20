from pathlib import Path

import nibabel as nib
import numpy as np

from voxelscout.ct_display import (
    DisplaySettings,
    automatic_window,
    inspect_hu,
    render_ct_slice,
)


def test_hu_inspection_keeps_distinct_outlier_counts() -> None:
    data = np.asarray([-2000.0, -1024.0, 0.0, 3071.0, 5000.0, np.nan])

    report = inspect_hu(data)

    assert report.finite_min == -2000
    assert report.finite_max == 5000
    assert report.nonfinite_count == 1
    assert report.below_clip_count == 1
    assert report.above_clip_count == 1


def test_automatic_window_is_robust_and_deterministic() -> None:
    data = np.linspace(-1000, 1000, 20_000, dtype=np.float32)
    data[0] = -1_000_000
    data[-1] = 1_000_000

    first = automatic_window(data)
    second = automatic_window(data)

    assert first == second
    assert -50 < first[0] < 50
    assert 1800 < first[1] < 2100


def test_display_transforms_do_not_mutate_hu_pixels() -> None:
    original = np.linspace(-1024, 1500, 64 * 64, dtype=np.float32).reshape(64, 64)
    original[0, 0] = 9999
    snapshot = original.copy()
    base = DisplaySettings(window_center=250, window_width=2000)

    plain = render_ct_slice(original, base)
    gamma = render_ct_slice(
        original,
        DisplaySettings(
            window_center=250,
            window_width=2000,
            transform="Gamma",
            gamma=0.7,
        ),
    )
    sigmoid_clahe = render_ct_slice(
        original,
        DisplaySettings(
            window_center=250,
            window_width=2000,
            transform="Sigmoid",
            clahe_enabled=True,
        ),
    )

    assert np.array_equal(original, snapshot)
    assert plain.dtype == gamma.dtype == sigmoid_clahe.dtype == np.uint8
    assert plain.shape == gamma.shape == sigmoid_clahe.shape == original.shape
    assert not np.array_equal(plain, gamma)
    assert not np.array_equal(plain, sigmoid_clahe)


def test_review_processing_does_not_modify_nifti_file(tmp_path: Path) -> None:
    path = tmp_path / "ct.nii.gz"
    data = np.arange(16 * 18 * 10, dtype=np.int16).reshape(16, 18, 10) - 1000
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    before = path.read_bytes()
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj)
    center, width = automatic_window(volume)

    render_ct_slice(
        volume[:, :, 5],
        DisplaySettings(
            window_center=center,
            window_width=width,
            transform="Gamma",
            clahe_enabled=True,
        ),
    )

    assert path.read_bytes() == before
