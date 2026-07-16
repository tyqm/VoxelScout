# VoxelScout

## A patient-facing spinal CT viewer

VoxelScout transforms medical imaging files that are difficult for patients to interpret into spinal views that are easier to browse, locate and explain.

> Current status: the first patient-facing GUI is implemented for local NIfTI CT volumes and trusted vertebra masks.

## Project aim

VoxelScout aims to make spinal CT data more accessible to patients with no prior medical imaging knowledge. The tool allows users to open a CT scan through a simple graphical interface and presents it in an understandable form, including familiar anatomical views, vertebral labels and an interactive visualisation of the visible spinal region.

The system supports understanding and communication rather than clinical diagnosis. It does not determine whether an abnormality is present and does not replace interpretation by a qualified healthcare professional.

## Problem statement

Patients may receive medical imaging data in formats such as NIfTI or DICOM, but these files are difficult to inspect without specialist software and prior knowledge of medical imaging. Existing research implementations primarily target researchers and clinicians, often requiring command-line tools, complex dependencies and technical configuration. VoxelScout investigates how spinal CT data can be presented through a simplified patient-facing interface.

## Product requirements

### Must be simple

- Start from a public example or upload one NIfTI volume.
- Present side, front and cross-sectional views with plain-language names.
- Offer useful CT display presets without requiring knowledge of window parameters.
- Allow a vertebra to be selected and located with one button.
- Explain that coloured labels show anatomy, not disease.

### Must be honest and safe

- Never infer a diagnosis.
- Never claim that a missing label means an abnormality.
- Display vertebra names only when a trusted segmentation is available.
- State clearly when automatic segmentation has not been run.
- Keep public demo data separate from patient data and exclude all scans from Git.

### Should remain lightweight

- Run locally on an ordinary computer for browsing.
- Build a decimated 3D surface only when requested.
- Retain the coarse-to-fine research path as a future way to reduce inference memory.

## Implemented GUI

The Streamlit application currently provides:

- automatic discovery of a local VerSe example;
- upload of a NIfTI CT volume and optional matching mask;
- sagittal (side), coronal (front) and axial (cross-sectional) sliders;
- bone, soft-tissue and lung display presets;
- adjustable label opacity;
- C1–C7, T1–T12, L1–L6 and T13 plain-language mappings;
- one-click navigation to a selected vertebra;
- an interactive 3D surface for the whole visible spine or one vertebra;
- patient-facing limitations and safety messages.

When no mask is available, VoxelScout still provides CT browsing but deliberately disables labels and 3D reconstruction. Automatic segmentation is a separate future milestone.

## Requirements and design

The stakeholder analysis, corrected terminology, CR1–CR12 traceability, safety boundaries and system architecture are maintained in [docs/system_design.md](docs/system_design.md).

The current GUI implements all requirements for trusted NIfTI CT/mask pairs. DICOM import and automatic segmentation remain explicitly planned extensions.

## Run locally

Activate the verified environment and update the project:

```powershell
conda activate voxelscout
git pull origin main
pip install -r requirements.txt
pip install -e .
pytest -q
```

Start the GUI:

```powershell
streamlit run app.py
```

The browser should open at:

```text
http://localhost:8501
```

For the existing VerSe validation data, choose **Use a local VerSe example**. VoxelScout will prefer `sub-gl017` when it is available.

## Data

Public source:

- VerSe repository: https://github.com/anjany/verse
- VerSe 2020 validation: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip

Expected local layout:

```text
data/raw/dataset-02validation/
├── rawdata/sub-gl017/sub-gl017_ct.nii.gz
└── derivatives/sub-gl017/sub-gl017_seg-vert_msk.nii.gz
```

Medical images, masks, model weights and generated outputs are excluded from Git.

## Revised development plan

### Phase 1 — Patient-facing viewer

- [x] Read and validate NIfTI CT/mask pairs.
- [x] Standardise orientation for display.
- [x] Inspect scan geometry and spacing variation.
- [x] Create orthogonal CT views and mask overlays.
- [x] Build the Streamlit GUI.
- [x] Add plain-language vertebra navigation.
- [x] Add optional lightweight 3D reconstruction.
- [ ] Conduct a short usability review with non-specialist users.
- [ ] Improve wording and interaction from observed confusion.

### Phase 2 — Input accessibility

- [ ] Add local DICOM-series import.
- [ ] Detect incomplete or mismatched series.
- [ ] Remove identifying metadata from any exported screenshots.
- [ ] Package the application for simpler local launch.

### Phase 3 — Automatic anatomical labelling

- [ ] Download the VerSe training set.
- [ ] Establish a reproducible 3D segmentation baseline.
- [ ] Integrate model inference behind the GUI.
- [ ] Mark predictions clearly as automated and potentially inaccurate.
- [ ] Compare prediction with trusted annotations using Dice and IoU.

### Phase 4 — Lightweight coarse-to-fine inference

- [ ] Localise the spine at 3 mm isotropic spacing.
- [ ] Segment the cropped ROI at 1 mm isotropic spacing.
- [ ] Compare memory, latency and segmentation quality with a single-stage model.
- [ ] Add failure-case analysis for thick-slice and unusual field-of-view scans.

## Evaluation

The project is evaluated as a patient-facing system, not only as a model:

- task completion: can a new user open a scan and locate a named vertebra?
- usability: number of steps, time to first useful view and points of confusion;
- correctness: spatial alignment, label mapping and preservation of mask classes;
- performance: load time, interaction latency, peak memory and 3D mesh size;
- model quality, when automatic segmentation is added: Dice, IoU and failure cases.

## Existing command-line tools

The earlier engineering tools remain available:

```powershell
voxelscout-inspect --image path\to\ct.nii.gz --label path\to\mask.nii.gz
voxelscout-inventory --dataset-root data\raw\dataset-02validation
voxelscout-preprocess --image path\to\ct.nii.gz --label path\to\mask.nii.gz --spacing 1 1 1
```

## Licensing and attribution

Project code is intended to use the MIT Licence. VerSe data is distributed separately under CC BY-SA 4.0. Follow the official dataset terms and cite the VerSe publications when reporting results.
