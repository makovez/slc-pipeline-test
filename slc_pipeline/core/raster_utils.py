from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


class RasterUtils:
    @staticmethod
    def same_grid(
        src_a: rasterio.io.DatasetReader,
        src_b: rasterio.io.DatasetReader,
        tol: float = 1e-12,
    ) -> bool:
        if src_a.crs != src_b.crs:
            return False
        if src_a.width != src_b.width or src_a.height != src_b.height:
            return False
        return np.allclose(tuple(src_a.transform), tuple(src_b.transform), atol=tol, rtol=0.0)

    @staticmethod
    def reproject_band_to_reference(
        src: rasterio.io.DatasetReader,
        ref: rasterio.io.DatasetReader,
        band_index: int = 1,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        dst = np.empty((ref.height, ref.width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, band_index),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=resampling,
        )
        return dst
