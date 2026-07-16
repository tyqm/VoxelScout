# VoxelScout

## Coarse-to-Fine 3D Object Localisation and Segmentation

VoxelScout is a learning and research project for building a coarse-to-fine 3D perception pipeline on publicly available volumetric CT data. The initial dataset is VerSe 2020.

> Status: the dataset inspection/preprocessing workflow and an integrated local
> spinal CT desktop viewer are available. Model training remains a later milestone.

## Target outcome

The project aims to:

- combine low-resolution object localisation with high-resolution ROI segmentation;
- implement intensity normalisation, voxel-spacing resampling, augmentation, and patch-based training with PyTorch and MONAI;
- compare 2D, single-stage 3D, and coarse-to-fine approaches using Dice, IoU, latency, and memory;
- visualize orthogonal slice overlays, confidence maps, and 3D surface reconstructions.

These are project objectives, not completed results. Results and measured metrics will be added as experiments are completed.

## Local environment

Install PyTorch separately so the CPU development installation does not overwrite a future CUDA installation:

```powershell
conda activate voxelscout
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Current verified local environment:

```text
PyTorch: 2.13.0+cpu
MONAI:   1.6.0
CUDA:    False
```

The local machine is for preprocessing, visualization, tests, and small CPU smoke runs. Full 3D training will use an NVIDIA GPU environment later.

## Launch the desktop viewer

```powershell
voxelscout-viewer
```

The viewer implements the current user-facing workflow in one local application:

- import a NIfTI CT file or a folder containing a DICOM series;
- automatically reorient the volume and apply a CT bone window for viewing;
- show linked sagittal, coronal, and axial slices;
- automatically match a VerSe-style `*_seg-vert_msk.nii.gz` companion label, or
  load another aligned NIfTI label manually;
- list, select, locate, and highlight individual labelled vertebrae;
- show a rotatable 3D vertebral surface (or a low-resolution bone preview when
  no vertebra mask is available);
- export the current labelled four-panel view to PNG or PDF.

You can also open a case directly:

```powershell
voxelscout-viewer --image "path\to\ct.nii.gz" --label "path\to\mask.nii.gz"
```

All image processing stays on the local machine. Viewer outputs are for viewing
and anatomical orientation only, not for clinical diagnosis. Reliable individual
vertebra names require an aligned segmentation label; an unlabelled CT is not
silently presented as an automatically diagnosed result.

## Public data

Use the restructured VerSe 2020 training and validation releases:

- Official repository: https://github.com/anjany/verse
- Training: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip
- Validation: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip

Do not commit CT volumes, masks, model weights, generated outputs, or proprietary company data.

Expected layout:

```text
data/raw/dataset-02validation/
├── rawdata/sub-verseXXX/
│   └── sub-verseXXX_ct.nii.gz
└── derivatives/sub-verseXXX/
    └── sub-verseXXX_seg-vert_msk.nii.gz
```

## Inspect one case

```powershell
voxelscout-inspect --image "path\to\ct.nii.gz" --label "path\to\mask.nii.gz" --output "outputs\first_case.png"
```

This validates spatial alignment, reports geometry and labels, and saves sagittal, coronal, and axial overlays.

## Build the dataset inventory

```powershell
voxelscout-inventory --dataset-root "data\raw\dataset-02validation" --output "outputs\verse20_validation_inventory.csv"
```

The generated patient-level CSV records paths, volume shape, voxel spacing, physical field of view, orientation, label count, label IDs, foreground size, and alignment status. The terminal also reports dataset-wide minimum, median, and maximum spacing.

## Development checks

```powershell
git pull
pip install -e .
pytest -q
```

## Roadmap

- [x] Create the repository
- [x] Configure the local Python environment
- [x] Add the data inspection command and synthetic test
- [x] Inspect one real VerSe CT/mask pair
- [x] Add the patient-level dataset inventory command
- [x] Add an integrated NIfTI/DICOM desktop viewer with orthogonal views
- [x] Add labelled vertebra navigation, highlighting, 3D surfaces, and export
- [ ] Analyse the validation inventory and select target spacing
- [ ] Download and split the training data
- [ ] Train a binary 3D U-Net baseline
- [ ] Add low-resolution localisation and ROI extraction
- [ ] Train the high-resolution fine segmentation model
- [ ] Compare 2D, 3D, and coarse-to-fine methods
- [ ] Add failure-case analysis

## Licensing and attribution

Project code will use the MIT License. VerSe data is distributed separately under CC BY-SA 4.0. Follow the official dataset terms and cite the VerSe publications when reporting results.
