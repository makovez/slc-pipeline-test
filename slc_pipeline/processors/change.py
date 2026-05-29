from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from scipy import signal

from slc_pipeline.core import RasterUtils


class ChangeDetectionProcessor:
    def process_coherence_change(
        self,
        new_coh: Path,
        old_coh: Path,
        output_tif: Path,
        output_ratio_tif: Path | None = None,
        eps: float = 1e-6,
    ) -> dict[str, str]:
        output_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(new_coh) as new_ds, rasterio.open(old_coh) as old_ds:
            new_arr = new_ds.read(1).astype(np.float32)

            if RasterUtils.same_grid(new_ds, old_ds):
                old_arr = old_ds.read(1).astype(np.float32)
            else:
                old_arr = RasterUtils.reproject_band_to_reference(old_ds, new_ds, band_index=1)
                print("Input grids differ: OLD coherence was reprojected to NEW grid.")

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

            with rasterio.open(output_tif, "w", **out_profile) as out_ds:
                out_ds.write(change, 1)

            outputs = {"change_tif": str(output_tif.resolve())}

            if output_ratio_tif is not None:
                ratio = np.full(new_arr.shape, np.nan, dtype=np.float32)
                ratio[valid] = new_arr[valid] / (old_arr[valid] + eps)
                with rasterio.open(output_ratio_tif, "w", **out_profile) as ratio_ds:
                    ratio_ds.write(ratio, 1)
                outputs["ratio_tif"] = str(output_ratio_tif.resolve())

        return outputs

    def process_polsar_change(
        self,
        input_c2_date_a: Path,
        input_c2_date_b: Path,
        output_dir: Path,
        boxcar_win: tuple[int, int] = (3, 3),
        low_pct: float = 2.0,
        high_pct: float = 98.0,
    ) -> dict[str, str]:
        if not input_c2_date_a.exists() or not input_c2_date_b.exists():
            missing = [
                str(p)
                for p in [input_c2_date_a, input_c2_date_b]
                if not p.exists()
            ]
            raise FileNotFoundError(f"Input files not found: {missing}")

        output_dir.mkdir(parents=True, exist_ok=True)

        metric_names = [
            "slc_pow1",
            "slc_pow2",
            "slc_dif1",
            "slc_dif2",
            "slc_tra1",
            "slc_tra2",
            "slc_wishart",
            "grd_difVV",
            "grd_difVH",
            "grd_ratioVV",
            "grd_ratioVH",
            "grd_NDifVV",
            "grd_NDifVH",
        ]

        with ExitStack() as stack:
            src_a = stack.enter_context(rasterio.open(input_c2_date_a))
            src_b = stack.enter_context(rasterio.open(input_c2_date_b))

            if src_a.count < 4 or src_b.count < 4:
                raise ValueError("C2 files must have at least 4 bands: C11, Re(C12), Im(C12), C22")

            out_profile = src_a.profile.copy()
            out_profile.update(dtype="float32", count=1, compress="lzw", nodata=None)
            outputs = self._open_output_datasets(stack, out_profile, metric_names, output_dir)

            b_aligned: dict[int, np.ndarray] | None = None
            if not RasterUtils.same_grid(src_a, src_b):
                dx = src_b.bounds.left - src_a.bounds.left
                dy = src_b.bounds.top - src_a.bounds.top
                px_x = dx / src_a.res[0]
                px_y = dy / abs(src_a.res[1])
                print(
                    "WARNING: C2 grids are not aligned. "
                    f"Auto-reprojecting date B onto date A grid (shift ~ {px_x:.2f}px, {px_y:.2f}px)."
                )
                b_aligned = {
                    1: RasterUtils.reproject_band_to_reference(src_b, src_a, 1),
                    2: RasterUtils.reproject_band_to_reference(src_b, src_a, 2),
                    3: RasterUtils.reproject_band_to_reference(src_b, src_a, 3),
                    4: RasterUtils.reproject_band_to_reference(src_b, src_a, 4),
                }

            c11_a = src_a.read(1)
            c12_re_a = src_a.read(2)
            c12_im_a = src_a.read(3)
            c22_a = src_a.read(4)
            c12_a = c12_re_a + 1j * c12_im_a

            if b_aligned is None:
                c11_b = src_b.read(1)
                c12_re_b = src_b.read(2)
                c12_im_b = src_b.read(3)
                c22_b = src_b.read(4)
            else:
                c11_b = b_aligned[1]
                c12_re_b = b_aligned[2]
                c12_im_b = b_aligned[3]
                c22_b = b_aligned[4]
            c12_b = c12_re_b + 1j * c12_im_b

            c11_a_f, c22_a_f, c12_a_f = self._boxcar_pol_filter_dual(c11_a, c22_a, c12_a, boxcar_win)
            c11_b_f, c22_b_f, c12_b_f = self._boxcar_pol_filter_dual(c11_b, c22_b, c12_b, boxcar_win)

            slc = self._slc_change_det_dual_window(c11_a_f, c22_a_f, c12_a_f, c11_b_f, c22_b_f, c12_b_f)
            grd = self._grd_change_det_dual_window(c11_a_f, c22_a_f, c11_b_f, c22_b_f)

            for name, arr in {**slc, **grd}.items():
                outputs[name].write(arr, 1)

        for name in metric_names:
            self._stretch_raster_to_percentile_window(output_dir / f"{name}.tif", low_pct=low_pct, high_pct=high_pct)

        return {name: str((output_dir / f"{name}.tif").resolve()) for name in metric_names}

    @staticmethod
    def _safe_div(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
        return numerator / (denominator + eps)

    @staticmethod
    def _boxcar_pol_filter_dual(
        t11_pre: np.ndarray,
        t22_pre: np.ndarray,
        t12_pre: np.ndarray,
        win: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        win1 = np.int16(win[0])
        win2 = np.int16(win[1])
        kernel = np.ones((win1, win2), np.float32) / (win1 * win2)

        t11 = signal.convolve2d(t11_pre, kernel, mode="same", boundary="fill", fillvalue=0)
        t22 = signal.convolve2d(t22_pre, kernel, mode="same", boundary="fill", fillvalue=0)
        t12 = signal.convolve2d(t12_pre, kernel, mode="same", boundary="fill", fillvalue=0)

        return t11, t22, t12

    def _slc_change_det_dual_window(
        self,
        c11_a: np.ndarray,
        c22_a: np.ndarray,
        c12_a: np.ndarray,
        c11_b: np.ndarray,
        c22_b: np.ndarray,
        c12_b: np.ndarray,
        eps: float = 1e-6,
    ) -> dict[str, np.ndarray]:
        c11_a = c11_a.astype(np.float32) + eps
        c22_a = c22_a.astype(np.float32) + eps
        c11_b = c11_b.astype(np.float32) + eps
        c22_b = c22_b.astype(np.float32) + eps
        c12_a = c12_a.astype(np.complex64)
        c12_b = c12_b.astype(np.complex64)

        det_cb = c11_b * c22_b - np.abs(c12_b) ** 2
        det_cb = det_cb + eps

        inv11 = c22_b / det_cb
        inv12 = -c12_b / det_cb
        inv21 = -np.conj(c12_b) / det_cb
        inv22 = c11_b / det_cb

        a11 = inv11 * c11_a + inv12 * np.conj(c12_a)
        a12 = inv11 * c12_a + inv12 * c22_a
        a21 = inv21 * c11_a + inv22 * np.conj(c12_a)
        a22 = inv21 * c12_a + inv22 * c22_a

        trace_a = a11 + a22
        det_a = a11 * a22 - a12 * a21

        delta = np.sqrt(trace_a * trace_a - 4.0 * det_a)
        eig1 = 0.5 * (trace_a + delta)
        eig2 = 0.5 * (trace_a - delta)

        eig_abs_1 = np.abs(eig1)
        eig_abs_2 = np.abs(eig2)

        pow1 = np.maximum(eig_abs_1, eig_abs_2)
        pow2 = 1.0 / (np.minimum(eig_abs_1, eig_abs_2) + eps)

        b11 = c11_a - c11_b
        b22 = c22_a - c22_b
        b12 = c12_a - c12_b

        herm_delta = np.sqrt((b11 - b22) ** 2 + 4.0 * np.abs(b12) ** 2)
        dif1 = 0.5 * (b11 + b22 + herm_delta)
        dif2 = 0.5 * (b11 + b22 - herm_delta)

        tra1 = np.real(trace_a)
        tra2 = np.real(self._safe_div(trace_a, det_a, eps))

        det_c1 = c11_a * c22_a - np.abs(c12_a) ** 2
        det_c2 = c11_b * c22_b - np.abs(c12_b) ** 2
        det_c = ((c11_a + c11_b) * (c22_a + c22_b) - np.abs(c12_a + c12_b) ** 2) / 2.0

        wishart = self._safe_div(np.abs(det_c1) * np.abs(det_c2), np.abs(det_c) ** 2, eps)

        return {
            "slc_pow1": np.real(pow1).astype(np.float32),
            "slc_pow2": np.real(pow2).astype(np.float32),
            "slc_dif1": np.real(dif1).astype(np.float32),
            "slc_dif2": np.real(dif2).astype(np.float32),
            "slc_tra1": tra1.astype(np.float32),
            "slc_tra2": tra2.astype(np.float32),
            "slc_wishart": np.real(wishart).astype(np.float32),
        }

    def _grd_change_det_dual_window(
        self,
        c11_a: np.ndarray,
        c22_a: np.ndarray,
        c11_b: np.ndarray,
        c22_b: np.ndarray,
        eps: float = 1e-6,
    ) -> dict[str, np.ndarray]:
        c11_a = c11_a.astype(np.float32)
        c22_a = c22_a.astype(np.float32)
        c11_b = c11_b.astype(np.float32)
        c22_b = c22_b.astype(np.float32)

        dif_vv = c11_a - c11_b
        dif_vh = c22_a - c22_b

        ratio_vv = self._safe_div(c11_a, c11_b, eps)
        ratio_vh = self._safe_div(c22_a, c22_b, eps)

        ndif_vv = self._safe_div(dif_vv, c11_a + c11_b, eps)
        ndif_vh = self._safe_div(dif_vh, c22_a + c22_b, eps)

        return {
            "grd_difVV": dif_vv.astype(np.float32),
            "grd_difVH": dif_vh.astype(np.float32),
            "grd_ratioVV": ratio_vv.astype(np.float32),
            "grd_ratioVH": ratio_vh.astype(np.float32),
            "grd_NDifVV": ndif_vv.astype(np.float32),
            "grd_NDifVH": ndif_vh.astype(np.float32),
        }

    @staticmethod
    def _open_output_datasets(
        stack: ExitStack,
        profile: dict,
        metric_names: list[str],
        output_dir: Path,
    ) -> dict[str, rasterio.io.DatasetWriter]:
        outputs: dict[str, rasterio.io.DatasetWriter] = {}
        for name in metric_names:
            out_path = output_dir / f"{name}.tif"
            outputs[name] = stack.enter_context(rasterio.open(out_path, "w", **profile))
        return outputs

    @staticmethod
    def _stretch_raster_to_percentile_window(path: Path, low_pct: float = 2.0, high_pct: float = 98.0) -> None:
        with rasterio.open(path, "r+") as ds:
            arr = ds.read(1)
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                return

            vals = arr[finite_mask]
            low = np.percentile(vals, low_pct)
            high = np.percentile(vals, high_pct)
            arr_clipped = arr.copy()
            arr_clipped[finite_mask] = np.clip(vals, low, high)
            ds.write(arr_clipped.astype(np.float32), 1)
