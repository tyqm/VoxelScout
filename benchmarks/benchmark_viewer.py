"""Benchmark the desktop app's header-only load and per-vertebra mesh build."""

from __future__ import annotations

import json
import time
from pathlib import Path

from voxelscout.desktop_data import load_segmented_case


def find_case() -> tuple[Path, Path]:
    image = Path(
        "data/raw/dataset-02validation/rawdata/sub-gl017/sub-gl017_ct.nii.gz"
    )
    mask = Path(
        "data/raw/dataset-02validation/derivatives/sub-gl017/"
        "sub-gl017_seg-vert_msk.nii.gz"
    )
    if image.is_file() and mask.is_file():
        return image, mask
    raise FileNotFoundError("The local sub-gl017 benchmark case was not found")


def main() -> None:
    image, mask = find_case()
    started = time.perf_counter()
    case = load_segmented_case(image, mask)
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    cached = load_segmented_case(image, mask)
    cached_ms = (time.perf_counter() - started) * 1000
    print(
        json.dumps(
            {
                "case": case.name,
                "shape": case.shape,
                "vertebrae": len(case.meshes),
                "first_build_seconds": round(first_seconds, 4),
                "cached_reload_ms": round(cached_ms, 4),
                "mesh_memory_mib": round(case.mesh_memory_mib, 2),
                "vertices": sum(len(mesh.vertices) for mesh in case.meshes),
                "faces": sum(len(mesh.faces) for mesh in case.meshes),
                "same_cached_object": cached is case,
                "ct_voxel_array_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
