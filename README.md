# VoxelScout

VoxelScout is a desktop educational tool that converts segmented spinal CT data
into an interactive 3D model, allowing non-specialist users to identify and
understand visible vertebrae through direct exploration.

Current workflow: `(standalone NIfTI CT or DICOM CT folder) → pretrained nnU-Net → labelled 3D spine`.

VoxelScout is the product and project name. `Lenx` is the short name retained for
its Windows desktop window; the title bar shows `VoxelScout — Lenx` to make that
relationship explicit. The current MVP version is 0.5.0.

## Desktop application

The Windows application deliberately focuses on one workflow:

1. Open a NIfTI CT or DICOM CT folder. VoxelScout uses a trusted companion mask
   when present and otherwise runs the configured nnU-Net backend.
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

For CT-only automatic inference:

```powershell
python app.py --ct path\to\scan_ct.nii.gz
```

For a DICOM CT folder, use the `Open CT` menu and choose `DICOM folder`, or run:

```powershell
python app.py --ct "D:\scans\dicom-series"
```

VoxelScout uses SimpleITK/GDCM to discover and order CT series. Localizers and
very short scout series are excluded when a diagnostic series is available. If
more than one candidate remains, the desktop app shows one compact chooser with
description, shape, and spacing. Slice dimensions, in-plane spacing, direction,
position, and inter-slice spacing are validated before conversion.

GDCM applies the DICOM rescale slope/intercept while reading, preserving CT HU.
VoxelScout writes a float32 NIfTI containing pixels and geometry only; it never
modifies the source DICOM or copies patient metadata into the cache. Converted
volumes are stored outside the repository under the local application-data cache
(override with `VOXELSCOUT_DICOM_CACHE_DIR`) and reuse the existing prediction
and mesh workflow. Loading reports `Reading DICOM series`, `Converting CT`, and
the existing prediction source/status stages.

## Segmentation modes

Trusted-mask mode remains the fastest and most reproducible path. Supply
`--mask`, or place a matching `*_seg-vert_msk.nii.gz` beside the CT or in the
official VerSe `derivatives/sub-*` location. A discovered trusted mask is always
used before automatic inference.

CT-only mode uses an external nnU-Net v2 command in the background. nnU-Net,
PyTorch, and model weights are intentionally not mandatory viewer dependencies.
VoxelScout automatically detects the adjacent development checkout used for the
verified local model (`VoxelScout-ML`) and its `verse-pretrained` Conda
environment. For another installation, configure the direct model folder and
the matching executable before starting VoxelScout:

```powershell
$env:VOXELSCOUT_NNUNET_MODEL_DIR = `
  "D:\models\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres"
$env:VOXELSCOUT_NNUNET_COMMAND = `
  "D:\miniforge3\envs\verse-pretrained\Scripts\nnUNetv2_predict_from_modelfolder.exe"
$env:VOXELSCOUT_NNUNET_FOLDS = "0"
$env:VOXELSCOUT_NNUNET_CHECKPOINT = "checkpoint_final.pth"
$env:VOXELSCOUT_NNUNET_DEVICE = "cpu"
python app.py --ct "D:\scans\scan_ct.nii.gz"
```

These settings match the verified pretrained VerSe model and its
`nnUNetv2_predict_from_modelfolder` command. This direct model-folder entry point
does not depend on a legacy dataset-ID or results-directory convention.
`VOXELSCOUT_CACHE_DIR` can relocate cached predictions. The model folder must
contain nnU-Net metadata and the selected fold checkpoint:

```text
nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres/
|-- dataset.json
|-- plans.json
`-- fold_0/checkpoint_final.pth
```

If the command, model directory, or checkpoint is unavailable, loading stops
with a specific error; VoxelScout never fabricates a mask or downloads a model.
Predictions are cached by CT modification state and backend/model configuration,
so changing the CT or configuration does not reuse a stale result. Prediction
label `26` is converted back to VerSe/VoxelScout label `28` before mesh creation.

The loading status identifies the selected source as `companion mask`, `cached
prediction`, or `real model inference`; this does not add configuration controls
to the desktop UI.

## Verified CT-only acceptance

The real local acceptance run copied a 512 x 512 x 229 CT into an independent
directory containing no mask, cleared the prediction cache, and opened only that
CT from the GUI. The pretrained nnU-Net process produced 11 labelled vertebrae,
the existing mesh pipeline rendered all 11, and hover/click selection was
confirmed in the 3D view. CPU inference took 224.87 seconds (about 225 seconds)
on the acceptance machine. Runtime is hardware-dependent; subsequent opens can
reuse the cached prediction.

## Performance design

- PySide6 provides a native Windows window; no browser or local web server opens.
- PyVistaQt/VTK performs camera interaction and actor picking in the viewport.
- DICOM is read and converted with SimpleITK/GDCM; no custom slice sorting is
  used.
- CT and segmentation geometry are validated before mesh construction; the CT
  is not resampled or normalized by VoxelScout.
- The compact integer segmentation is released after mesh construction.
- Marching Cubes runs separately on each vertebra's tight bounding box with a
  reduced sampling step.
- Oversized meshes are simplified with topology-preserving VTK decimation.
- Meshes are cached in memory by file path, modification time, and quality.
- Hover and click operate on existing VTK actors and never rescan the mask.
- Loading and mesh generation run in a background QThread with progress updates.

## Display-only CT review

After a case loads, `Review CT` opens a separate axial before/after window. This
tool reads the existing CT but never writes it and is not connected to the
nnU-Net command, prediction cache, segmentation, or mesh inputs.

The default review uses deterministic 0.5th/99.5th percentile automatic
windowing after display-only clipping to `-1024…3071 HU`. The window follows the
DICOM VOI concept and can be disabled for manual center/width values. Gamma,
Sigmoid, and slice-wise CLAHE are separate optional transforms applied only
after values have been mapped to `[0, 1]`; all are off by default. The panel also
reports non-finite samples and values outside the configured HU clip range.

- DICOM windowing definition: https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html
- scikit-image exposure transforms: https://scikit-image.org/docs/stable/api/skimage.exposure.html

Run the real-case benchmark with:

```powershell
python benchmarks/benchmark_viewer.py
```

On the included 512 x 512 x 229 validation case (11 vertebrae), the current
implementation builds the meshes in about 12.7 seconds, keeps about 2.27 MiB of
mesh data, and reloads the same in-memory case in about 3 ms. Results vary by
machine; the benchmark prints the local measurements.

## Current limitation

Automatic segmentation requires a separately installed and configured
pretrained nnU-Net model. NIfTI and DICOM CT input are supported; diagnostic or
abnormality detection is not supported.

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

Medical images, cached predictions, masks, model weights, generated screenshots,
and patient data are excluded from Git.

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
