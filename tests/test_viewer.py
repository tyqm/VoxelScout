import numpy as np

from voxelscout.anatomy import vertebra_info
from voxelscout.viewer import (
    export_contact_sheet,
    label_centroid,
    label_options,
    render_slice,
    surface_from_mask,
    window_ct,
)


def test_patient_friendly_anatomy_names() -> None:
    assert vertebra_info(1).code == "C1"
    assert vertebra_info(8).code == "T1"
    assert vertebra_info(20).code == "L1"
    assert vertebra_info(28).code == "T13"


def test_render_and_locate_label() -> None:
    image = np.zeros((20, 18, 16), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[6:12, 5:11, 4:10] = 20

    rendered = render_slice(
        image,
        mask,
        axis=2,
        index=7,
        centre=0,
        width=1000,
        opacity=0.5,
    )

    assert rendered.shape == (18, 20, 3)
    assert rendered.dtype == np.uint8
    assert label_centroid(mask, 20) == (8, 8, 6)
    assert list(label_options(mask).values()) == [20]


def test_window_and_surface() -> None:
    volume = np.asarray([-1000.0, 0.0, 1000.0])
    result = window_ct(volume, centre=0, width=2000)
    assert np.allclose(result, [0.0, 0.5, 1.0])

    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 4:12, 4:12] = 1
    vertices, faces = surface_from_mask(mask, (1, 1, 1), step_size=1)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3


def test_export_contact_sheet_is_png() -> None:
    first = np.zeros((20, 30, 3), dtype=np.uint8)
    second = np.full((30, 20, 3), 180, dtype=np.uint8)
    exported = export_contact_sheet(
        [first, second],
        ["Sagittal — slice 10", "Axial — slice 12"],
        panel_size=64,
    )
    assert exported.startswith(b"\\x89PNG\\r\\n\\x1a\\n")
    assert len(exported) > 100
