from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import signal

# Hardcoded input paths requested by the user.
INPUT_C2_DATE_A = Path("otuput_emilia_20230323/work/c2_subset_bbox.tif")
INPUT_C2_DATE_B = Path("outputs_frame1_20230522_path95_desc/work/c2.tif")
OUTPUT_DIR = Path("output_change_det_emilia")
BOXCAR_WIN = (3, 3)


def _safe_div(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
	return numerator / (denominator + eps)

def Boxcar_Pol_filter_Dual(T11_pre, T22_pre, T12_pre, win):
#    win = [7, 7]
    win1 = np.int16(win[0])
    win2 = np.int16(win[1])
    
    # Create the kernel of the boxcar
    kernel  = np.ones((win1,win2),np.float32)/(win1*win2)
    
    # Filter the images using convolve2d
    T11 =  signal.convolve2d(T11_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)
    T22 =  signal.convolve2d(T22_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)
    T12 =  signal.convolve2d(T12_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)


    return T11, T22, T12


def _slc_change_det_dual_window(
	c11_a: np.ndarray,
	c22_a: np.ndarray,
	c12_a: np.ndarray,
	c11_b: np.ndarray,
	c22_b: np.ndarray,
	c12_b: np.ndarray,
	eps: float = 1e-6,
) -> dict[str, np.ndarray]:
	# PRE acquisition
	c11_a = c11_a.astype(np.float32) + eps # VV
	c22_a = c22_a.astype(np.float32) + eps # VH
	
    # POST acquisition
	c11_b = c11_b.astype(np.float32) + eps # VV
	c22_b = c22_b.astype(np.float32) + eps # VH

    # Cross-pol complex correlation
	c12_a = c12_a.astype(np.complex64)
	c12_b = c12_b.astype(np.complex64)

    # Inverse of second covariance (Cb) matrix, faster than np.linalg.inv
	det_cb = c11_b * c22_b - np.abs(c12_b) ** 2 # determinant of Cb
	det_cb = det_cb + eps

	inv11 = c22_b / det_cb # inverse of Cb matrix
	inv12 = -c12_b / det_cb # note: not conjugating c12_b here because we will conjugate it when multiplying by Ca, so it is more efficient to do it here once instead of twice later
	inv21 = -np.conj(c12_b) / det_cb # note: conjugating c12_b here because it will be multiplied by Ca which is not conjugated, so we need to conjugate it here to ensure the correct Hermitian structure of the inverse matrix
	inv22 = c11_b / det_cb # inverse of Cb matrix

    # Matrix multiplication of A = Cb^-1 * Ca (manual implementation for 2x2 matrices, faster than np.matmul for small matrices)
	# this replaces np.matmul(invCb, Ca) and is equivalent to invCb @ Ca
	a11 = inv11 * c11_a + inv12 * np.conj(c12_a)
	a12 = inv11 * c12_a + inv12 * c22_a
	a21 = inv21 * c11_a + inv22 * np.conj(c12_a)
	a22 = inv21 * c12_a + inv22 * c22_a


    # Eigenvalue-based change metrics
	# Instead of computing the full eigen decomposition, we can compute the eigenvalues of a 2x2 matrix using the closed-form solution, which is much faster than np.linalg.eig for small matrices.
	# Formula: [Tr(A) +- sqrt(Tr(A)^2 - 4*det(A))] / 2
	trace_a = a11 + a22
	det_a = a11 * a22 - a12 * a21

	delta = np.sqrt(trace_a * trace_a - 4.0 * det_a)
	eig1 = 0.5 * (trace_a + delta)
	eig2 = 0.5 * (trace_a - delta)

    # Power change metrics based on eigenvalues
	eig_abs_1 = np.abs(eig1)
	eig_abs_2 = np.abs(eig2)

    # deviation from 1 = change
	pow1 = np.maximum(eig_abs_1, eig_abs_2) # If pow1 is much greater than 1 - strong increase in some scattering mechanism
	pow2 = 1.0 / (np.minimum(eig_abs_1, eig_abs_2) + eps) # If pow2 is much greater than 1 - strong decrease in some scattering mechanism


    # Difference-based change metrics based on eigenvalues of the difference matrix B = Ca - Cb
	b11 = c11_a - c11_b
	b22 = c22_a - c22_b
	b12 = c12_a - c12_b

    # Difference eigenvalues 
    # Eigenvalues of B can be computed using the same closed-form solution for 2x2 matrices, which is much faster than np.linalg.eig(B) for small matrices.
	# Formula: [ B11 + B22 +- sqrt((B11 - B22)^2 + 4*|B12|^2) ] / 2
	herm_delta = np.sqrt((b11 - b22) ** 2 + 4.0 * np.abs(b12) ** 2)
	dif1 = 0.5 * (b11 + b22 + herm_delta)
	dif2 = 0.5 * (b11 + b22 - herm_delta)

    # Trace, (sum of eigenvalues). Proportional to total scattering power, so it is a measure of overall change in scattering power
	tra1 = np.real(trace_a)
	tra2 = np.real(_safe_div(trace_a, det_a, eps))

     # (proportional to) Wishart 
    # Measures how likely both matrices come from same distribution
    # we need the determinant of the images
	det_c1 = c11_a * c22_a - np.abs(c12_a) ** 2
	det_c2 = c11_b * c22_b - np.abs(c12_b) ** 2
	det_c = ((c11_a + c11_b) * (c22_a + c22_b) - np.abs(c12_a + c12_b) ** 2) / 2.0

	wishart = _safe_div(np.abs(det_c1) * np.abs(det_c2), np.abs(det_c) ** 2, eps)

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

	ratio_vv = _safe_div(c11_a, c11_b, eps)
	ratio_vh = _safe_div(c22_a, c22_b, eps)

	ndif_vv = _safe_div(dif_vv, c11_a + c11_b, eps)
	ndif_vh = _safe_div(dif_vh, c22_a + c22_b, eps)

	return {
		"grd_difVV": dif_vv.astype(np.float32),
		"grd_difVH": dif_vh.astype(np.float32),
		"grd_ratioVV": ratio_vv.astype(np.float32),
		"grd_ratioVH": ratio_vh.astype(np.float32),
		"grd_NDifVV": ndif_vv.astype(np.float32),
		"grd_NDifVH": ndif_vh.astype(np.float32),
	}


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
		print(f"Stretched {path.name}: p{low_pct}={low:.6g}, p{high_pct}={high:.6g}")


def _same_grid(src_a: rasterio.io.DatasetReader, src_b: rasterio.io.DatasetReader, tol: float = 1e-12) -> bool:
	if src_a.crs != src_b.crs:
		return False
	if src_a.width != src_b.width or src_a.height != src_b.height:
		return False
	return np.allclose(tuple(src_a.transform), tuple(src_b.transform), atol=tol, rtol=0.0)


def _reproject_band_to_reference(
	src: rasterio.io.DatasetReader,
	ref: rasterio.io.DatasetReader,
	band_index: int,
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


def process_change_detection() -> None:
	if not INPUT_C2_DATE_A.exists() or not INPUT_C2_DATE_B.exists():
		missing = [str(p) for p in [INPUT_C2_DATE_A, INPUT_C2_DATE_B] if not p.exists()]
		raise FileNotFoundError(f"Input files not found: {missing}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	with ExitStack() as stack:
		src_a = stack.enter_context(rasterio.open(INPUT_C2_DATE_A))
		src_b = stack.enter_context(rasterio.open(INPUT_C2_DATE_B))

		if src_a.count < 4 or src_b.count < 4:
			raise ValueError("C2 files must have at least 4 bands: C11, Re(C12), Im(C12), C22")

		out_profile = src_a.profile.copy()
		out_profile.update(dtype="float32", count=1, compress="lzw", nodata=None)

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
		outputs = _open_output_datasets(stack, out_profile, metric_names, OUTPUT_DIR)

		b_aligned: dict[int, np.ndarray] | None = None
		if not _same_grid(src_a, src_b):
			dx = src_b.bounds.left - src_a.bounds.left
			dy = src_b.bounds.top - src_a.bounds.top
			px_x = dx / src_a.res[0]
			px_y = dy / abs(src_a.res[1])
			print(
				"WARNING: C2 grids are not aligned. "
				f"Auto-reprojecting date B onto date A grid "
				f"(shift ~ {px_x:.2f}px, {px_y:.2f}px)."
			)
			b_aligned = {
				1: _reproject_band_to_reference(src_b, src_a, 1),
				2: _reproject_band_to_reference(src_b, src_a, 2),
				3: _reproject_band_to_reference(src_b, src_a, 3),
				4: _reproject_band_to_reference(src_b, src_a, 4),
			}

		# Read full C2 bands and apply 3x3 boxcar before any change metric.
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

		c11_a_f, c22_a_f, c12_a_f = Boxcar_Pol_filter_Dual(c11_a, c22_a, c12_a, BOXCAR_WIN)
		c11_b_f, c22_b_f, c12_b_f = Boxcar_Pol_filter_Dual(c11_b, c22_b, c12_b, BOXCAR_WIN)

		slc = _slc_change_det_dual_window(c11_a_f, c22_a_f, c12_a_f, c11_b_f, c22_b_f, c12_b_f)
		grd = _grd_change_det_dual_window(c11_a_f, c22_a_f, c11_b_f, c22_b_f)

		for name, arr in {**slc, **grd}.items():
			outputs[name].write(arr, 1)

	for name in metric_names:
		_stretch_raster_to_percentile_window(OUTPUT_DIR / f"{name}.tif", low_pct=2.0, high_pct=98.0)

	print(f"Change detection completed. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
	process_change_detection()
