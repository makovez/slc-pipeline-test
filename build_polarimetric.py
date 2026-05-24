import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

from slc_processing import (
    BBox,
    build_polarimetric_graph,
    clip_geotiff_to_bbox,
    detect_gpt_path,
    infer_c2_band_indices,
    overlapping_burst_range_for_bbox,
    raster_has_valid_values,
    run_gpt,
    write_graph,
)


def write_single_band(src: rasterio.DatasetReader, arr: np.ndarray, out_tif: Path) -> None:
    meta = src.meta.copy()
    meta.update(dtype='float32', count=1, compress='deflate', nodata=np.nan)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


def export_c2_components(c2_tif: Path, c2_components_dir: Path) -> None:
    with rasterio.open(c2_tif) as src:
        c11_i, c12r_i, c12i_i, c22_i = infer_c2_band_indices(src.descriptions)

        c11 = src.read(c11_i + 1).astype(np.float32)
        c12_real = src.read(c12r_i + 1).astype(np.float32)
        c12_imag = src.read(c12i_i + 1).astype(np.float32)
        c22 = src.read(c22_i + 1).astype(np.float32)

        c12 = np.sqrt(np.maximum(c12_real * c12_real + c12_imag * c12_imag, 0.0)).astype(np.float32)

        write_single_band(src, c11, c2_components_dir / 'c11.tif')
        write_single_band(src, c12_real, c2_components_dir / 'c12_real.tif')
        write_single_band(src, c12_imag, c2_components_dir / 'c12_imag.tif')
        write_single_band(src, c12, c2_components_dir / 'c12.tif')
        write_single_band(src, c22, c2_components_dir / 'c22.tif')


def c2_has_signal(c2_tif: Path) -> bool:
    with rasterio.open(c2_tif) as src:
        for i in range(1, src.count + 1):
            arr = src.read(i)
            finite = np.isfinite(arr)
            if not finite.any():
                continue
            if np.count_nonzero(arr[finite]) > 0:
                return True
    return False


def merge_c2_products(src_tifs: list[Path], dst_tif: Path) -> None:
    if not src_tifs:
        raise ValueError('Nessun C2 da fondere.')

    src_datasets = [rasterio.open(p) for p in src_tifs]
    try:
        merged_arr, merged_transform = rio_merge(src_datasets, nodata=np.nan)
        meta = src_datasets[0].meta.copy()
        meta.update(
            {
                'height': merged_arr.shape[1],
                'width': merged_arr.shape[2],
                'transform': merged_transform,
                'count': merged_arr.shape[0],
                'dtype': 'float32',
                'compress': 'deflate',
                'nodata': np.nan,
            }
        )
    finally:
        for ds in src_datasets:
            ds.close()

    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, 'w', **meta) as dst:
        dst.write(merged_arr.astype(np.float32))


def _load_config(path: Path) -> dict:
    raw = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        return json.loads(raw)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Config YAML richiesto ma PyYAML non installato. Usa JSON o installa pyyaml.') from exc
    return yaml.safe_load(raw)


def _parse_bbox(value: object) -> BBox:
    if isinstance(value, list) and len(value) == 4:
        return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    if isinstance(value, dict):
        return BBox(
            min_lon=float(value['min_lon']),
            min_lat=float(value['min_lat']),
            max_lon=float(value['max_lon']),
            max_lat=float(value['max_lat']),
        )
    raise ValueError('bbox non valido. Usa lista [min_lon,min_lat,max_lon,max_lat] o dict.')


def _scene_date_from_zip_name(path: Path) -> str:
    m = re.search(r'_(\d{8})T\d{6}_', path.name)
    if m:
        return m.group(1)
    return 'unknown_date'


def run_build_polarimetric(
    input_zip: Path,
    bbox: BBox,
    out_dir: Path,
    subswath_candidates: list[str] | None = None,
    gpt_override: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / 'work'
    work_dir.mkdir(parents=True, exist_ok=True)

    gpt_path = Path(gpt_override) if gpt_override else detect_gpt_path()
    print(f'SNAP GPT: {gpt_path}')

    if subswath_candidates is None:
        subswath_candidates = ['IW1', 'IW2', 'IW3']

    c2_per_swath: list[Path] = []
    selected_subswaths: list[str] = []

    for subswath in subswath_candidates:
        print(f'Valuto {subswath}...')
        try:
            fb, lb = overlapping_burst_range_for_bbox(input_zip, subswath, bbox)
            print(f'  overlap burst: {fb}-{lb}')
        except Exception as exc:
            print(f'  {subswath} scartato: {exc}')
            continue

        c2_swath = work_dir / f'c2_{subswath}.tif'
        graph_xml = work_dir / f'graph_polarimetric_{subswath}.xml'

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
            run_gpt(gpt_path, graph_xml, extra_args=['-Dsnap.parallelism=1'])
            if raster_has_valid_values(c2_swath):
                selected_subswaths.append(subswath)
                c2_per_swath.append(c2_swath)
                print(f'  {subswath} aggiunto alla fusione')
            else:
                print(f'  {subswath} prodotto senza valori validi')
        except Exception as exc:
            print(f'  {subswath} fallito in GPT: {exc}')

    if not c2_per_swath:
        raise RuntimeError('Nessun subswath valido trovato (IW1/IW2/IW3).')

    print(f'Subswath selezionate: {", ".join(selected_subswaths)}')

    c2_merged = work_dir / 'c2_merged.tif'
    c2_subset = work_dir / 'c2_subset_bbox.tif'

    if len(c2_per_swath) == 1:
        c2_merged = c2_per_swath[0]
    else:
        merge_c2_products(c2_per_swath, c2_merged)

    clip_geotiff_to_bbox(c2_merged, c2_subset, bbox, nodata=np.nan)

    if not c2_has_signal(c2_subset):
        raise RuntimeError(
            'C2 prodotto ma senza segnale (tutte bande a zero nel bbox). '
            'Prova un bbox piu ampio/diverso o una coppia di scene differente.'
        )

    c2_components_dir = out_dir / 'c2_components'
    export_c2_components(c2_subset, c2_components_dir)

    result = {
        'input_zip': str(input_zip.resolve()),
        'scene_date': _scene_date_from_zip_name(input_zip),
        'subswaths': selected_subswaths,
        'c2_subset': str(c2_subset.resolve()),
        'c2_components_dir': str(c2_components_dir.resolve()),
        'c11': str((c2_components_dir / 'c11.tif').resolve()),
        'c12_real': str((c2_components_dir / 'c12_real.tif').resolve()),
        'c12_imag': str((c2_components_dir / 'c12_imag.tif').resolve()),
        'c12': str((c2_components_dir / 'c12.tif').resolve()),
        'c22': str((c2_components_dir / 'c22.tif').resolve()),
    }
    (out_dir / 'polarimetric_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('\nOutput generati:')
    print(result['c11'])
    print(result['c12_real'])
    print(result['c12_imag'])
    print(result['c12'])
    print(result['c22'])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build C2 e componenti polarimetriche da una SLC ZIP')
    parser.add_argument('--config', help='Percorso config JSON/YAML (opzionale)')
    parser.add_argument('--input-zip', help='Path ZIP SLC input')
    parser.add_argument('--output-dir', help='Directory output polarimetrico')
    parser.add_argument('--bbox', help='bbox CSV: min_lon,min_lat,max_lon,max_lat')
    parser.add_argument('--gpt-path', help='Override path eseguibile SNAP gpt')
    return parser.parse_args()


def _bbox_from_csv(csv_value: str) -> BBox:
    parts = [float(p.strip()) for p in csv_value.split(',')]
    if len(parts) != 4:
        raise ValueError('bbox CSV non valido, usa: min_lon,min_lat,max_lon,max_lat')
    return BBox(parts[0], parts[1], parts[2], parts[3])


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = _load_config(Path(args.config).expanduser().resolve())
        bbox = _parse_bbox(cfg['bbox'])
        processing = cfg.get('processing', {})
        subs = processing.get('subswaths', ['IW1', 'IW2', 'IW3'])
        gpt_cfg = cfg.get('snap', {}).get('gpt_path')
        input_zip = Path(args.input_zip or cfg.get('input_zip', ''))
        output_dir = Path(args.output_dir or cfg.get('output_dir', ''))
    else:
        if not (args.input_zip and args.output_dir and args.bbox):
            raise RuntimeError('Senza --config servono --input-zip, --output-dir e --bbox.')
        bbox = _bbox_from_csv(args.bbox)
        subs = ['IW1', 'IW2', 'IW3']
        gpt_cfg = args.gpt_path
        input_zip = Path(args.input_zip)
        output_dir = Path(args.output_dir)

    if not input_zip.exists():
        raise FileNotFoundError(f'ZIP input non trovato: {input_zip}')

    run_build_polarimetric(
        input_zip=input_zip.expanduser().resolve(),
        bbox=bbox,
        out_dir=output_dir.expanduser().resolve(),
        subswath_candidates=[str(s) for s in subs],
        gpt_override=gpt_cfg,
    )


if __name__ == '__main__':
    main()
