#!/usr/bin/env python3
"""Simple change detection from coherence rasters.

Computes:
    intcoh_20260501_20260407 - intcoh_20230323_20230522
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

NEW_COH = Path("output_coherence_emilia_20260501_20260407/intcoh_20260501_20260407.tif")
OLD_COH = Path("output_coherence_emilia_20260313_20260406/intcoh_20260313_20260406.tif")
OUTPUT_DIR = Path("output_coherence_change_emilia_20260313_20260406_minus_20260501_20260407")
OUTPUT_TIF = OUTPUT_DIR / "intcoh_change_20260501_20260407_minus_20260313_20260406.tif"


def _same_grid(src_a: rasterio.io.DatasetReader, src_b: rasterio.io.DatasetReader, tol: float = 1e-12) -> bool:
    if src_a.crs != src_b.crs:
        return False
    if src_a.width != src_b.width or src_a.height != src_b.height:
        return False
    return np.allclose(tuple(src_a.transform), tuple(src_b.transform), atol=tol, rtol=0.0)


def _reproject_to_reference(
    src: rasterio.io.DatasetReader,
    ref: rasterio.io.DatasetReader,
    band_index: int = 1,
) -> np.ndarray:
    dst = np.empty((ref.height, ref.width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, band_index),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref.transform,
        dst_crs=ref.crs,
        resampling=Resampling.bilinear,
    )
    return dst


def main() -> None:
    if not NEW_COH.exists() or not OLD_COH.exists():
        missing = [str(p) for p in [NEW_COH, OLD_COH] if not p.exists()]
        raise FileNotFoundError(
            "Input coherence file(s) not found: "
            f"{missing}. Place the two TIFF files in the workspace root or edit paths in the script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(NEW_COH) as new_ds, rasterio.open(OLD_COH) as old_ds:
        new_arr = new_ds.read(1).astype(np.float32)

        if _same_grid(new_ds, old_ds):
            old_arr = old_ds.read(1).astype(np.float32)
        else:
            old_arr = _reproject_to_reference(old_ds, new_ds, band_index=1)
            print("Input grids differ: OLD was reprojected/resampled to NEW grid.")

        nodata_new = new_ds.nodata
        nodata_old = old_ds.nodata

        valid = np.isfinite(new_arr) & np.isfinite(old_arr)
        if nodata_new is not None:
            valid &= new_arr != nodata_new
        if nodata_old is not None:
            valid &= old_arr != nodata_old

        change = np.full(new_arr.shape, np.nan, dtype=np.float32)
        change[valid] = new_arr[valid] - old_arr[valid]

        out_profile = new_ds.profile.copy()
        out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

        with rasterio.open(OUTPUT_TIF, "w", **out_profile) as out_ds:
            out_ds.write(change, 1)

    print(f"Saved: {OUTPUT_TIF}")


if __name__ == "__main__":
    main()
