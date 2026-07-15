# VoxelScout

## Coarse-to-Fine 3D Object Localisation and Segmentation

VoxelScout is a learning and research project for building a coarse-to-fine 3D perception pipeline on publicly available volumetric CT data. The initial dataset is VerSe 2020.

> Status: environment and repository scaffold complete; data inspection is the next milestone.

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

Verify:

```powershell
python -c "import torch, monai; print(torch.__version__); print(monai.__version__); print(torch.cuda.is_available())"
```

Current verified local environment:

```text
PyTorch: 2.13.0+cpu
MONAI:   1.6.0
CUDA:    False
```

The local machine is for preprocessing, visualization, tests, and small CPU smoke runs. Full 3D training will use an NVIDIA GPU environment later.

## Public data

Use the restructured VerSe 2020 training and validation releases:

- Official repository: https://github.com/anjany/verse
- Training: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip
- Validation: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip

Do not commit CT volumes, masks, model weights, generated outputs, or proprietary company data.

Expected example:

```text
data/verse20/
├── rawdata/sub-verseXXX/
│   └── sub-verseXXX_dir-orient_ct.nii.gz
└── derivatives/sub-verseXXX/
    └── sub-verseXXX_dir-orient_seg-vert_msk.nii.gz
```

## First workflow

Pull the current repository and install it in editable mode:

```powershell
git pull
pip install -e .
pytest
```

After placing one matching CT/mask pair under `data/verse20`:

```powershell
voxelscout-inspect --image "path\to\ct.nii.gz" --label "path\to\mask.nii.gz" --output "outputs\first_case.png"
```

The command:

1. reorients CT and mask to a canonical orientation;
2. checks their shapes and affine transforms;
3. reports voxel spacing, CT range, mask labels, and foreground size;
4. saves sagittal, coronal, and axial CT/mask overlays.

## Roadmap

- [x] Create the repository
- [x] Configure the local Python environment
- [x] Add the data inspection command and synthetic test
- [ ] Inspect one real VerSe CT/mask pair
- [ ] Build a patient-level train/validation manifest
- [ ] Train a binary 3D U-Net baseline
- [ ] Add low-resolution localisation and ROI extraction
- [ ] Train the high-resolution fine segmentation model
- [ ] Compare 2D, 3D, and coarse-to-fine methods
- [ ] Add failure-case analysis and 3D visualization

## Licensing and attribution

Project code will use the MIT License. VerSe data is distributed separately under CC BY-SA 4.0. Follow the official dataset terms and cite the VerSe publications when reporting results.
