from pathlib import Path
import shutil

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from slc_processing import (
    BBox,
    build_interferometric_coherence_graph,
    clip_geotiff_to_bbox,
    detect_gpt_path,
    raster_has_valid_values,
    run_gpt,
    write_graph,
)

master_zip = Path(
    "Emilia/"
    "S1C_IW_SLC__1SDV_20260407T170527_20260407T170555_007113_00E67E_436C.zip"
)
slave_zip = Path(
    "Emilia/"
    "S1C_IW_SLC__1SDV_20260501T170528_20260501T170556_007463_00F256_C6B9.zip"
)
out_dir = Path("output_coherence_emilia_20230323_20230522")
work_dir = out_dir / "work"
out_dir.mkdir(parents=True, exist_ok=True)
work_dir.mkdir(parents=True, exist_ok=True)

bbox = BBox(
    min_lon=11.5891752243041992,
    min_lat=44.3745611959093438,
    max_lon=12.1745191687090415,
    max_lat=44.6658172607421875,
)

dem_name = "SRTM 1Sec HGT"
pixel_spacing_m = 13.92
subswath_candidates = ["IW1", "IW2", "IW3"]


def merge_single_band_products(src_tifs: list[Path], dst_tif: Path) -> None:
    if not src_tifs:
        raise ValueError("Nessun raster da fondere.")

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


def main() -> None:
    gpt_path = detect_gpt_path()
    print(f"SNAP GPT: {gpt_path}")

    coh_per_swath: list[Path] = []
    selected_subswaths: list[str] = []

    for subswath in subswath_candidates:
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
                    dem_name=dem_name,
                    pixel_spacing_m=pixel_spacing_m,
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
        raise RuntimeError("Nessun subswath valido trovato (IW1/IW2/IW3).")

    print(f"Subswath selezionate: {', '.join(selected_subswaths)}")

    out_final = out_dir / "intcoh_20230323_20230522.tif"
    if len(coh_per_swath) == 1:
        shutil.copyfile(coh_per_swath[0], out_final)
    else:
        merge_single_band_products(coh_per_swath, out_final)

    print("\nOutput generato:")
    print(out_final)


if __name__ == "__main__":
    main()
