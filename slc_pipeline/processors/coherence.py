from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from slc_pipeline.config import BBox, PipelineConfig
from slc_pipeline.core.snap_ops import (
    build_interferometric_coherence_graph,
    clip_geotiff_to_bbox,
    detect_gpt_path,
    raster_has_valid_values,
    run_gpt,
    write_graph,
)


class InterferometricCoherenceProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def _merge_single_band_products(src_tifs: list[Path], dst_tif: Path) -> None:
        if not src_tifs:
            raise ValueError("Nessun raster da fondere")

        src_datasets = [rasterio.open(p) for p in src_tifs]
        try:
            merged_arr, merged_transform = rio_merge(src_datasets, nodata=np.nan)
            meta = src_datasets[0].meta.copy()
            meta.update(
                {
                    "height": merged_arr.shape[1],
                    "width": merged_arr.shape[2],
                    "transform": merged_transform,
                    "count": 1,
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
            dst.write(merged_arr[0].astype(np.float32), 1)

    def process(self, master_zip: Path, slave_zip: Path, out_dir: Path) -> dict[str, str | list[str]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = out_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        gpt_path = Path(self.config.snap.gpt_path) if self.config.snap.gpt_path else detect_gpt_path()
        bbox: BBox = self.config.bbox

        coh_per_swath: list[Path] = []
        selected_subswaths: list[str] = []

        for subswath in self.config.snap.subswaths:
            print(f"Valuto {subswath}...")

            coh_raw = work_dir / f"intcoh_{subswath}.tif"
            coh_clip = work_dir / f"intcoh_{subswath}_bbox.tif"
            graph_xml = work_dir / f"graph_intcoh_{subswath}.xml"

            try:
                write_graph(
                    build_interferometric_coherence_graph(
                        master_zip=master_zip,
                        slave_zip=slave_zip,
                        out_tif=coh_raw,
                        bbox=bbox,
                        dem_name=self.config.snap.dem_name,
                        pixel_spacing_m=self.config.snap.pixel_spacing_m,
                        subswath=subswath,
                    ),
                    graph_xml,
                )
                run_gpt(gpt_path, graph_xml)
                clip_geotiff_to_bbox(coh_raw, coh_clip, bbox, nodata=np.nan)
                if raster_has_valid_values(coh_clip):
                    selected_subswaths.append(subswath)
                    coh_per_swath.append(coh_clip)
                    print(f"  {subswath} aggiunto alla fusione")
                else:
                    print(f"  {subswath} prodotto senza valori validi")
            except Exception as exc:
                print(f"  {subswath} fallito in GPT: {exc}")

        if not coh_per_swath:
            raise RuntimeError("Nessun subswath valido trovato (IW1/IW2/IW3)")

        stem = f"intcoh_{master_zip.stem}_{slave_zip.stem}"[:180]
        out_final = out_dir / f"{stem}.tif"
        if len(coh_per_swath) == 1:
            shutil.copyfile(coh_per_swath[0], out_final)
        else:
            self._merge_single_band_products(coh_per_swath, out_final)

        return {
            "output_tif": str(out_final.resolve()),
            "selected_subswaths": selected_subswaths,
            "master_zip": str(master_zip.resolve()),
            "slave_zip": str(slave_zip.resolve()),
        }
