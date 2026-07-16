"""VoxelScout patient-facing Streamlit application."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from voxelscout.anatomy import vertebra_info
from voxelscout.viewer import (
    export_contact_sheet,
    label_centroid,
    label_options,
    load_volume,
    render_slice,
    surface_from_mask,
    validate_mask,
)


st.set_page_config(
    page_title="VoxelScout",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_local_nifti(path: str, modified_time: float):
    del modified_time
    return load_volume(Path(path))


@st.cache_data(show_spinner=False)
def load_uploaded_nifti(contents: bytes, filename: str):
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(contents)
            temporary_path = temporary.name
        return load_volume(Path(temporary_path))
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def find_demo_pair() -> tuple[Path, Path] | None:
    candidates = sorted(
        Path("data/raw").glob("dataset-*/rawdata/sub-*/*_ct.nii.gz"),
        key=lambda path: ("sub-gl017" not in str(path), str(path)),
    )
    for image_path in candidates:
        dataset_root = image_path.parents[2]
        subject = image_path.parent.name
        masks = sorted(
            (dataset_root / "derivatives" / subject).glob("*_seg-vert_msk.nii.gz")
        )
        if len(masks) == 1:
            return image_path, masks[0]
    return None


def load_inputs():
    st.sidebar.header("Open a scan")
    source = st.sidebar.radio(
        "Choose a simple starting point",
        ("Use a local VerSe example", "Upload my NIfTI file"),
    )

    if source == "Use a local VerSe example":
        pair = find_demo_pair()
        if pair is None:
            st.info(
                "No local VerSe example was found. Put the public dataset under "
                "data/raw, or choose Upload my NIfTI file."
            )
            st.stop()
        image_path, mask_path = pair
        st.sidebar.success(f"Example loaded: {image_path.parent.name}")
        image, image_affine, spacing = load_local_nifti(
            str(image_path), image_path.stat().st_mtime
        )
        mask, mask_affine, _ = load_local_nifti(
            str(mask_path), mask_path.stat().st_mtime
        )
        case_name = image_path.parent.name
        return image, image_affine, spacing, mask, mask_affine, case_name

    image_file = st.sidebar.file_uploader(
        "CT volume (.nii or .nii.gz)",
        type=("nii", "gz"),
        help="The scan is processed locally in this running application.",
    )
    with st.sidebar.expander("Optional vertebra labels"):
        mask_file = st.file_uploader(
            "Segmentation mask (.nii or .nii.gz)",
            type=("nii", "gz"),
            help=(
                "A matching mask enables vertebra names and 3D views. "
                "VoxelScout does not invent labels when no segmentation is available."
            ),
        )
    if image_file is None:
        st.info("Upload a NIfTI CT volume to begin.")
        st.stop()

    image, image_affine, spacing = load_uploaded_nifti(
        image_file.getvalue(), image_file.name
    )
    mask = mask_affine = None
    if mask_file is not None:
        mask, mask_affine, _ = load_uploaded_nifti(
            mask_file.getvalue(), mask_file.name
        )
    return image, image_affine, spacing, mask, mask_affine, image_file.name


st.title("VoxelScout")
st.caption("A simpler way to explore and locate structures in a spinal CT scan")
st.warning(
    "VoxelScout supports understanding and communication only. It does not detect "
    "disease, decide whether an abnormality is present, or replace a report from a "
    "qualified healthcare professional.",
    icon="⚠️",
)

with st.spinner("Preparing the scan…"):
    image, image_affine, spacing, mask, mask_affine, case_name = load_inputs()

if image.ndim != 3:
    st.error(f"VoxelScout currently supports one 3D CT volume; received {image.shape}.")
    st.stop()

if mask is not None:
    try:
        validate_mask(image, image_affine, mask, mask_affine)
        mask = np.rint(mask).astype(np.int16)
    except ValueError as error:
        st.error(f"The optional mask cannot be used: {error}")
        mask = None

st.sidebar.divider()
st.sidebar.header("Display")
window_preset = st.sidebar.selectbox(
    "CT view",
    ("Bone detail", "Soft tissue", "Lung", "Custom"),
)
presets = {
    "Bone detail": (300, 1800),
    "Soft tissue": (40, 400),
    "Lung": (-600, 1500),
}
if window_preset == "Custom":
    window_centre = st.sidebar.slider("Window centre", -1000, 1500, 300, 10)
    window_width = st.sidebar.slider("Window width", 100, 4000, 1800, 50)
else:
    window_centre, window_width = presets[window_preset]
    st.sidebar.caption(f"Centre {window_centre} HU · width {window_width} HU")

opacity = st.sidebar.slider(
    "Label overlay",
    min_value=0.0,
    max_value=0.9,
    value=0.45 if mask is not None else 0.0,
    step=0.05,
    disabled=mask is None,
)

labels = label_options(mask)
selected_label = None
highlight_selected = False
if labels:
    st.sidebar.divider()
    st.sidebar.header("Find a vertebra")
    selected_name = st.sidebar.selectbox("Vertebra", tuple(labels))
    selected_label = labels[selected_name]
    highlight_selected = st.sidebar.toggle(
        "Highlight only this vertebra",
        value=False,
        help="Temporarily hides the other coloured labels to make one vertebra easier to discuss.",
    )

display_mask = mask
if mask is not None and highlight_selected and selected_label is not None:
    display_mask = np.where(mask == selected_label, mask, 0)

case_key = f"{case_name}-{image.shape}"
slider_keys = {
    0: f"sagittal-{case_key}",
    1: f"coronal-{case_key}",
    2: f"axial-{case_key}",
}
default_indices = {axis: image.shape[axis] // 2 for axis in range(3)}
for axis, key in slider_keys.items():
    if key not in st.session_state or st.session_state[key] >= image.shape[axis]:
        st.session_state[key] = default_indices[axis]

if selected_label is not None and st.sidebar.button(
    "Locate this vertebra", type="primary", use_container_width=True
):
    centre = label_centroid(mask, selected_label)
    for axis in range(3):
        st.session_state[slider_keys[axis]] = centre[axis]

explore_tab, guide_tab, surface_tab, about_tab = st.tabs(
    ("Explore scan", "Spine guide", "3D view", "About this tool")
)

with explore_tab:
    info_columns = st.columns(4)
    info_columns[0].metric("Scan", case_name)
    info_columns[1].metric("Volume", " × ".join(str(v) for v in image.shape))
    info_columns[2].metric(
        "Voxel spacing",
        " × ".join(f"{value:.2f}" for value in spacing) + " mm",
    )
    info_columns[3].metric("Visible labels", len(labels) if labels else "Not supplied")

    names = {
        0: ("Sagittal", "Side view"),
        1: ("Coronal", "Front view"),
        2: ("Axial", "Cross-section"),
    }
    columns = st.columns(3)
    export_views = []
    export_titles = []
    for axis, column in enumerate(columns):
        title, explanation = names[axis]
        with column:
            st.subheader(title)
            st.caption(explanation)
            index = st.slider(
                f"{title} slice",
                0,
                image.shape[axis] - 1,
                key=slider_keys[axis],
            )
            rendered = render_slice(
                image,
                display_mask,
                axis=axis,
                index=index,
                centre=window_centre,
                width=window_width,
                opacity=opacity,
            )
            st.image(rendered, use_container_width=True)
            export_views.append(rendered)
            export_titles.append(f"{title} — slice {index}")

    export_png = export_contact_sheet(export_views, export_titles)
    st.download_button(
        "Export current labelled views",
        data=export_png,
        file_name="VoxelScout_spine_views.png",
        mime="image/png",
        use_container_width=True,
        help="Exports only the displayed images and view names; source-file metadata are not embedded.",
    )
    st.caption(
        "The exported PNG contains the current views only. Check it before sharing "
        "and follow the privacy advice provided by your healthcare service."
    )

    if mask is None:
        st.info(
            "This CT can be browsed, but no vertebra labels are available. "
            "Automatic segmentation is a later VoxelScout milestone; this version "
            "only displays labels supplied by a trusted mask."
        )

with guide_tab:
    st.header("What is visible?")
    if not labels:
        st.info("Add a matching vertebra mask to show the plain-language spine guide.")
    else:
        selected_labels = sorted(set(labels.values()))
        rows = []
        for label in selected_labels:
            info = vertebra_info(label)
            voxels = int(np.count_nonzero(mask == label))
            rows.append(
                {
                    "Vertebra": info.code,
                    "Spinal region": info.region,
                    "Plain-language location": info.plain_location,
                    "Labelled voxels": f"{voxels:,}",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(
            "The coloured areas show labelled anatomy, not a diagnosis. "
            "A label does not indicate that anything is wrong."
        )

with surface_tab:
    st.header("Interactive 3D spine")
    if mask is None:
        st.info("A segmentation mask is required to construct the 3D surface.")
    else:
        scope = st.radio(
            "Surface",
            ("All visible vertebrae", "Selected vertebra"),
            horizontal=True,
        )
        surface_label = selected_label if scope == "Selected vertebra" else None
        if st.button("Build 3D view", type="primary"):
            with st.spinner("Building a lightweight surface…"):
                vertices, faces = surface_from_mask(
                    mask,
                    spacing,
                    label=surface_label,
                    step_size=2,
                )
            figure = go.Figure(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color="#2B7A78",
                        opacity=1.0,
                        flatshading=False,
                    )
                ]
            )
            figure.update_layout(
                height=650,
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    aspectmode="data",
                    xaxis_title="Left–right (mm)",
                    yaxis_title="Front–back (mm)",
                    zaxis_title="Head–foot (mm)",
                ),
            )
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                "This surface is reconstructed from the supplied segmentation "
                "and is intended for orientation and communication."
            )

with about_tab:
    st.header("What VoxelScout does")
    st.markdown(
        """
        **Intended users**

        - **Patients:** browse a scan and locate anatomy discussed by a clinician.
        - **Medical students:** relate CT slices to labelled three-dimensional anatomy.
        - **Technical users new to medical imaging:** learn common inputs, outputs,
          anatomical views and vertebral labels before moving to specialist software.

        VoxelScout turns specialist volumetric files into a view that is easier to
        browse, locate and explain. It standardises orientation for display, provides
        familiar anatomical views, and translates numeric vertebra labels into
        plain-language locations.

        **Current version**

        - Reads local NIfTI CT volumes.
        - Shows synchronised sagittal, coronal and axial views.
        - Provides bone, soft-tissue and lung display presets.
        - Uses an optional trusted mask for vertebra navigation and 3D reconstruction.

        **Not included yet**

        - DICOM folder import.
        - Automatic vertebra segmentation for a new patient scan.
        - Detection of fractures, tumours or any other abnormality.
        - Medical advice or interpretation.

        **Short glossary**

        - **CT:** Computed Tomography, an X-ray-based three-dimensional scan.
        - **DICOM:** the common clinical format for a series of medical images.
        - **NIfTI:** a file format that stores a complete three-dimensional volume.
        - **Sagittal / coronal / axial:** side, front and cross-sectional views.
        - **Voxel spacing:** the physical size represented by one 3D image element.
        - **RAS:** Right–Anterior–Superior, an orientation convention used to
          standardise how a volume is displayed.
        """
    )
