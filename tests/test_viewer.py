import numpy as np

from voxelscout.anatomy import vertebra_info
from voxelscout.viewer import (
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
