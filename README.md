# VoxelScout

VoxelScout is a desktop educational tool that converts segmented spinal CT data
into an interactive 3D model, allowing non-specialist users to identify and
understand visible vertebrae through direct exploration.

## Desktop application

The Windows application deliberately focuses on one workflow:

1. Open a NIfTI CT and its matching VerSe segmentation mask.
2. Wait while one simplified mesh is generated and cached for each vertebra.
3. Rotate, zoom, and pan the coloured 3D spine.
4. Hover to see the vertebra code, anatomical name, and spinal region.
5. Click a vertebra to keep it highlighted and show a short explanation.
6. Export the current 3D camera view as a PNG.

## Run

```powershell
conda activate voxelscout
pip install -r requirements.txt
pip install -e .
voxelscout-viewer
```

You can also launch it directly:

```powershell
python app.py
```

For a scripted launch with a known VerSe pair:

```powershell
python app.py --ct path\to\scan_ct.nii.gz --mask path\to\scan_seg-vert_msk.nii.gz
```

## Performance design

- PySide6 provides a native Windows window; no browser or local web server opens.
- PyVistaQt/VTK performs camera interaction and actor picking in the viewport.
- The CT voxel array is not read for the default 3D workflow; only its header is
  used to validate shape, affine, spacing, and orientation.
- The compact integer segmentation is released after mesh construction.
- Marching Cubes runs separately on each vertebra's tight bounding box with a
  reduced sampling step.
- Oversized meshes are simplified with topology-preserving VTK decimation.
- Meshes are cached in memory by file path, modification time, and quality.
- Hover and click operate on existing VTK actors and never rescan the mask.
- Loading and mesh generation run in a background QThread with progress updates.

Run the real-case benchmark with:

```powershell
python benchmarks/benchmark_viewer.py
```

On the included 512 x 512 x 229 validation case (11 vertebrae), the current
implementation builds the meshes in about 12.7 seconds, keeps about 2.27 MiB of
mesh data, and reloads the same in-memory case in about 3 ms. Results vary by
machine; the benchmark prints the local measurements.

## Current limitation

VoxelScout does not yet segment an arbitrary CT. A trusted matching mask is
required. Integrating a pretrained vertebra segmentation model is future work;
the current public demonstration uses VerSe CT/mask pairs.

The application does not diagnose fractures, tumours, or other abnormalities and
does not replace interpretation by a qualified healthcare professional.

## Public data

- VerSe repository: https://github.com/anjany/verse
- VerSe 2020 validation: https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip

Expected local layout:

```text
data/raw/dataset-02validation/
├── rawdata/sub-gl017/sub-gl017_ct.nii.gz
└── derivatives/sub-gl017/sub-gl017_seg-vert_msk.nii.gz
```

Medical images, masks, model weights, generated screenshots, and patient data are
excluded from Git.

## Prepare VerSe 2020 for nnU-Net v2

VoxelScout includes a deterministic dataset preparation command. It copies CT
files without reorientation, resampling, normalization, or recompression. It
preserves vertebra identities and only remaps official VerSe label `28` (T13) to
consecutive training label `26`. The official validation cases remain outside
`imagesTr` as a true holdout set.

Install the project itself; nnU-Net, MONAI, and PyTorch are not required for this
step:

```powershell
Set-Location "C:\path\to\VoxelScout"
python -m pip install -e .
```

Validate and preview every planned case without writing files:

```powershell
voxelscout-prepare-nnunet `
  --training-root "D:\datasets\verse20training" `
  --validation-root "D:\datasets\verse20validation" `
  --nnunet-raw "D:\nnUNet\nnUNet_raw" `
  --holdout-output "D:\nnUNet\verse20_holdout" `
  --dry-run
```

Run the real preparation after the dry run succeeds:

```powershell
voxelscout-prepare-nnunet `
  --training-root "D:\datasets\verse20training" `
  --validation-root "D:\datasets\verse20validation" `
  --nnunet-raw "D:\nnUNet\nnUNet_raw" `
  --holdout-output "D:\nnUNet\verse20_holdout"
```

The validation argument is optional. When supplied, its subjects are checked
against the training subjects and written only to the separate holdout tree.
The command produces:

```text
D:\nnUNet\nnUNet_raw\Dataset501_VerSe20\
|-- imagesTr\VerSe20_CASE_0000.nii.gz
|-- labelsTr\VerSe20_CASE.nii.gz
|-- dataset.json
|-- label_mapping.json
|-- manifest.csv
`-- preparation_summary.json

D:\nnUNet\verse20_holdout\
|-- images\VerSe20_CASE_0000.nii.gz
`-- labels\VerSe20_CASE.nii.gz
```

`label_mapping.json` contains both VerSe-to-training and
training-to-VerSe mappings, including prediction label `26` back to VoxelScout
label `28`. Existing compatible files are reused. Incompatible CTs, labels, or
metadata are never silently overwritten. This command prepares data only; it
does not run nnU-Net planning, preprocessing, or training.

## Research tools

The older preprocessing experiments remain optional and are not exposed in the
desktop GUI. Install their heavier dependencies only when needed:

```powershell
pip install -e ".[research]"
```

## Licensing

Project code uses the MIT Licence. VerSe data is distributed separately under
CC BY-SA 4.0.
