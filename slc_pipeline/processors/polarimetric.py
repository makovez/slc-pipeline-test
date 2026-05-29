from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from slc_pipeline.config import BBox, PipelineConfig
from slc_pipeline.core.snap_ops import (
    build_polarimetric_graph,
    clip_geotiff_to_bbox,
    detect_gpt_path,
    infer_c2_band_indices,
    overlapping_burst_range_for_bbox,
    raster_has_valid_values,
    run_gpt,
    write_graph,
)


class PolarimetricProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def _write_single_band(src: rasterio.DatasetReader, arr: np.ndarray, out_tif: Path) -> None:
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, compress="deflate", nodata=np.nan)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(arr.astype(np.float32), 1)

    def _export_c2_components(self, c2_tif: Path, c2_components_dir: Path) -> None:
        with rasterio.open(c2_tif) as src:
            c11_i, c12r_i, c12i_i, c22_i = infer_c2_band_indices(src.descriptions)

            c11 = src.read(c11_i + 1).astype(np.float32)
            c12_real = src.read(c12r_i + 1).astype(np.float32)
            c12_imag = src.read(c12i_i + 1).astype(np.float32)
            c22 = src.read(c22_i + 1).astype(np.float32)

            c12 = np.sqrt(np.maximum(c12_real * c12_real + c12_imag * c12_imag, 0.0)).astype(np.float32)

            self._write_single_band(src, c11, c2_components_dir / "c11.tif")
            self._write_single_band(src, c12_real, c2_components_dir / "c12_real.tif")
            self._write_single_band(src, c12_imag, c2_components_dir / "c12_imag.tif")
            self._write_single_band(src, c12, c2_components_dir / "c12.tif")
            self._write_single_band(src, c22, c2_components_dir / "c22.tif")

    @staticmethod
    def _c2_has_signal(c2_tif: Path) -> bool:
        with rasterio.open(c2_tif) as src:
            for i in range(1, src.count + 1):
                arr = src.read(i)
                finite = np.isfinite(arr)
                if not finite.any():
                    continue
                if np.count_nonzero(arr[finite]) > 0:
                    return True
        return False

    @staticmethod
    def _merge_c2_products(src_tifs: list[Path], dst_tif: Path) -> None:
        if not src_tifs:
            raise ValueError("Nessun C2 da fondere")

        src_datasets = [rasterio.open(p) for p in src_tifs]
        try:
            merged_arr, merged_transform = rio_merge(src_datasets, nodata=np.nan)
            meta = src_datasets[0].meta.copy()
            meta.update(
                {
                    "height": merged_arr.shape[1],
                    "width": merged_arr.shape[2],
                    "transform": merged_transform,
                    "count": merged_arr.shape[0],
                    "dtype": "float32",
                    "compress": "deflate",
                    "nodata": np.nan,
                }
            )
        finally:
            for ds in src_datasets:
                ds.close()

        dst_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_tif, "w", **meta) as dst:
            dst.write(merged_arr.astype(np.float32))

    def process(self, input_zip: Path, out_dir: Path) -> dict[str, str | list[str]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = out_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        gpt_path = Path(self.config.snap.gpt_path) if self.config.snap.gpt_path else detect_gpt_path()
        bbox: BBox = self.config.bbox

        c2_per_swath: list[Path] = []
        selected_subswaths: list[str] = []

        for subswath in self.config.snap.subswaths:
            print(f"Valuto {subswath}...")
            try:
                fb, lb = overlapping_burst_range_for_bbox(input_zip, subswath, bbox)
                print(f"  overlap burst: {fb}-{lb}")
            except Exception as exc:
                print(f"  {subswath} scartato: {exc}")
                continue

            c2_swath = work_dir / f"c2_{subswath}.tif"
            graph_xml = work_dir / f"graph_polarimetric_{subswath}.xml"

            try:
                write_graph(
                    build_polarimetric_graph(
                        input_zip=input_zip,
                        out_tif=c2_swath,
                        bbox=bbox,
                        subswath=subswath,
                    ),
                    graph_xml,
                )
                run_gpt(gpt_path, graph_xml, extra_args=["-Dsnap.parallelism=1"])
                if raster_has_valid_values(c2_swath):
                    selected_subswaths.append(subswath)
                    c2_per_swath.append(c2_swath)
                    print(f"  {subswath} aggiunto alla fusione")
                else:
                    print(f"  {subswath} prodotto senza valori validi")
            except Exception as exc:
                print(f"  {subswath} fallito in GPT: {exc}")

        if not c2_per_swath:
            raise RuntimeError("Nessun subswath valido trovato (IW1/IW2/IW3)")

        c2_merged = work_dir / "c2_merged.tif"
        c2_subset = work_dir / "c2_subset_bbox.tif"

        if len(c2_per_swath) == 1:
            c2_merged = c2_per_swath[0]
        else:
            self._merge_c2_products(c2_per_swath, c2_merged)

        clip_geotiff_to_bbox(c2_merged, c2_subset, bbox, nodata=np.nan)

        if not self._c2_has_signal(c2_subset):
            raise RuntimeError(
                "C2 prodotto ma senza segnale (tutte bande a zero nel bbox). "
                "Prova un bbox piu ampio/diverso o una scena differente."
            )

        c2_components_dir = out_dir / "c2_components"
        self._export_c2_components(c2_subset, c2_components_dir)

        result = {
            "input_zip": str(input_zip.resolve()),
            "subswaths": selected_subswaths,
            "c2_subset": str(c2_subset.resolve()),
            "c2_components_dir": str(c2_components_dir.resolve()),
            "c11": str((c2_components_dir / "c11.tif").resolve()),
            "c12_real": str((c2_components_dir / "c12_real.tif").resolve()),
            "c12_imag": str((c2_components_dir / "c12_imag.tif").resolve()),
            "c12": str((c2_components_dir / "c12.tif").resolve()),
            "c22": str((c2_components_dir / "c22.tif").resolve()),
        }
        (out_dir / "polarimetric_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
