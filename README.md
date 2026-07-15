# VoxelScout
Developed a coarse-to-fine 3D perception pipeline on publicly available volumetric CT data, combining low-resolution object localisation with high-resolution ROI segmentation.

Implemented preprocessing for volumetric data, including intensity normalisation, voxel-spacing resampling, data augmentation and patch-based training using PyTorch and MONAI.

Benchmarked 2D, 3D and coarse-to-fine segmentation approaches using Dice, IoU, inference latency and memory consumption, and conducted failure-case analysis across varying scan geometries.

Built an interactive visualisation pipeline for orthogonal slice overlays, confidence maps and 3D surface reconstruction.
