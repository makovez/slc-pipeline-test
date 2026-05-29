from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

from slc_pipeline.config import PipelineConfig
from slc_pipeline.download.asf_downloader import ASFDownloader
from slc_pipeline.processors import (
    ChangeDetectionProcessor,
    InterferometricCoherenceProcessor,
    PolarimetricProcessor,
)


def _cmd_download(config_path: Path) -> int:
    cfg = PipelineConfig.from_file(config_path)
    downloader = ASFDownloader(cfg)
    selection = downloader.run_download_triplet()

    summary = {
        "event_scene_name": selection.event_scene_name,
        "event_start_utc": selection.event_start_utc,
        "reference_scene_names": selection.reference_scene_names,
        "output_dir": selection.output_dir,
    }

    out_file = Path(selection.output_dir) / "download_selection.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDownload completato.")
    print(json.dumps(summary, indent=2))
    print(f"Riepilogo salvato in: {out_file}")
    return 0


def _cmd_changedet_coh(new_coh: Path, old_coh: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = ChangeDetectionProcessor()
    outputs = proc.process_coherence_change(
        new_coh=new_coh,
        old_coh=old_coh,
        output_tif=output_dir / "intcoh_change.tif",
        output_ratio_tif=output_dir / "intcoh_ratio.tif",
    )
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_changedet_polsar(c2_a: Path, c2_b: Path, output_dir: Path, boxcar: tuple[int, int]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = ChangeDetectionProcessor()
    outputs = proc.process_polsar_change(
        input_c2_date_a=c2_a,
        input_c2_date_b=c2_b,
        output_dir=output_dir,
        boxcar_win=boxcar,
    )
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_process_coherence(config_path: Path, master_zip: Path, slave_zip: Path, output_dir: Path) -> int:
    cfg = PipelineConfig.from_file(config_path)
    proc = InterferometricCoherenceProcessor(cfg)
    outputs = proc.process(master_zip=master_zip, slave_zip=slave_zip, out_dir=output_dir)
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_process_polarimetric(config_path: Path, input_zip: Path, output_dir: Path) -> int:
    cfg = PipelineConfig.from_file(config_path)
    proc = PolarimetricProcessor(cfg)
    outputs = proc.process(input_zip=input_zip, out_dir=output_dir)
    print(json.dumps(outputs, indent=2))
    return 0


def _resolve_scene_zip(download_dir: Path, scene_name: str) -> Path:
    candidates = sorted(download_dir.glob(f"{scene_name}*.zip"))
    if candidates:
        return candidates[0]
    candidates = sorted(download_dir.glob(f"*{scene_name}*.zip"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"ZIP non trovato per scena {scene_name} in {download_dir}")


def _load_or_create_selection(cfg: PipelineConfig, selection_file: Path) -> dict[str, Any]:
    if selection_file.exists():
        return json.loads(selection_file.read_text(encoding="utf-8"))

    downloader = ASFDownloader(cfg)
    selection = downloader.run_download_triplet()
    summary = {
        "event_scene_name": selection.event_scene_name,
        "event_start_utc": selection.event_start_utc,
        "reference_scene_names": selection.reference_scene_names,
        "output_dir": selection.output_dir,
    }
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _cmd_run_parallel_workflow(config_path: Path, selection_file: Path | None, output_dir: Path) -> int:
    cfg = PipelineConfig.from_file(config_path)
    download_dir = cfg.download.output_dir
    selection_path = selection_file or (download_dir / "download_selection.json")
    selection = _load_or_create_selection(cfg, selection_path)

    event_scene = selection["event_scene_name"]
    ref_scenes = selection["reference_scene_names"]
    if len(ref_scenes) < 2:
        raise RuntimeError("Servono 2 scene di riferimento nel selection file")

    # Ordered by downloader: first ref is newer (2nd image), second ref is older (3rd image).
    ref2_scene = ref_scenes[0]
    ref3_scene = ref_scenes[1]

    event_zip = _resolve_scene_zip(download_dir, event_scene)
    ref2_zip = _resolve_scene_zip(download_dir, ref2_scene)
    ref3_zip = _resolve_scene_zip(download_dir, ref3_scene)

    output_dir.mkdir(parents=True, exist_ok=True)
    coh_new_dir = output_dir / "coherence_1_event_2_ref"
    coh_old_dir = output_dir / "coherence_2_ref_3_ref"
    pol_event_dir = output_dir / "polarimetric_1_event"
    pol_ref2_dir = output_dir / "polarimetric_2_ref"
    pol_change_dir = output_dir / "change_polsar_1_vs_2"
    coh_change_dir = output_dir / "change_coherence_new_vs_old"

    print("Step 1/2: coerenze e polarimetric in parallelo...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_new = ex.submit(
            InterferometricCoherenceProcessor(cfg).process,
            master_zip=ref2_zip,
            slave_zip=event_zip,
            out_dir=coh_new_dir,
        )
        f_old = ex.submit(
            InterferometricCoherenceProcessor(cfg).process,
            master_zip=ref3_zip,
            slave_zip=ref2_zip,
            out_dir=coh_old_dir,
        )
        f_pol_event = ex.submit(
            PolarimetricProcessor(cfg).process,
            input_zip=event_zip,
            out_dir=pol_event_dir,
        )
        f_pol_ref2 = ex.submit(
            PolarimetricProcessor(cfg).process,
            input_zip=ref2_zip,
            out_dir=pol_ref2_dir,
        )

        coh_new = f_new.result()
        coh_old = f_old.result()
        pol_event = f_pol_event.result()
        pol_ref2 = f_pol_ref2.result()

    print("Step 2/2: change detection in parallelo (PolSAR e coherence)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_cd_pol = ex.submit(
            ChangeDetectionProcessor().process_polsar_change,
            input_c2_date_a=Path(str(pol_event["c2_subset"])),
            input_c2_date_b=Path(str(pol_ref2["c2_subset"])),
            output_dir=pol_change_dir,
        )
        f_cd_coh = ex.submit(
            ChangeDetectionProcessor().process_coherence_change,
            new_coh=Path(str(coh_new["output_tif"])),
            old_coh=Path(str(coh_old["output_tif"])),
            output_tif=coh_change_dir / "intcoh_change.tif",
            output_ratio_tif=coh_change_dir / "intcoh_ratio.tif",
        )
        pol_change = f_cd_pol.result()
        coh_change = f_cd_coh.result()

    summary = {
        "input": {
            "event_scene": event_scene,
            "reference_scene_2": ref2_scene,
            "reference_scene_3": ref3_scene,
            "event_zip": str(event_zip.resolve()),
            "reference_2_zip": str(ref2_zip.resolve()),
            "reference_3_zip": str(ref3_zip.resolve()),
        },
        "coherence": {
            "new_coh_1_vs_2": coh_new,
            "old_coh_2_vs_3": coh_old,
            "change": coh_change,
        },
        "polsar": {
            "event_1": pol_event,
            "reference_2": pol_ref2,
            "change_input_a_c2": str(pol_event["c2_subset"]),
            "change_input_b_c2": str(pol_ref2["c2_subset"]),
            "change_1_vs_2": pol_change,
            "slc_pow2": pol_change.get("slc_pow2"),
        },
    }

    out_json = output_dir / "parallel_workflow_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Workflow completato.")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {out_json}")
    return 0


def _parse_boxcar(value: str) -> tuple[int, int]:
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Valore boxcar non valido, usa ad esempio 3,3")
    return parts[0], parts[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLC pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Scarica 3 scene SLC (evento flood + 2 riferimenti)")
    dl.add_argument(
        "--config",
        default="config.yaml",
        help="Percorso config YAML/JSON (default: config.yaml)",
    )

    coh = sub.add_parser("changedet-coh", help="Change detection su rasters di coerenza")
    coh.add_argument("--new-coh", required=True, help="Raster coerenza recente")
    coh.add_argument("--old-coh", required=True, help="Raster coerenza storico")
    coh.add_argument("--output-dir", required=True, help="Directory output")

    pol = sub.add_parser("changedet-polsar", help="Change detection da due raster C2")
    pol.add_argument("--input-c2-a", required=True, help="Raster C2 data A")
    pol.add_argument("--input-c2-b", required=True, help="Raster C2 data B")
    pol.add_argument("--output-dir", required=True, help="Directory output")
    pol.add_argument("--boxcar", default="3,3", help="Finestra boxcar rows,cols")

    coh_proc = sub.add_parser("process-coherence", help="Genera coerenza interferometrica da due SLC ZIP")
    coh_proc.add_argument("--config", default="config.yaml", help="Config YAML/JSON")
    coh_proc.add_argument("--master-zip", required=True, help="ZIP master")
    coh_proc.add_argument("--slave-zip", required=True, help="ZIP slave")
    coh_proc.add_argument("--output-dir", required=True, help="Directory output")

    pol_proc = sub.add_parser("process-polarimetric", help="Genera C2 e componenti polarimetriche")
    pol_proc.add_argument("--config", default="config.yaml", help="Config YAML/JSON")
    pol_proc.add_argument("--input-zip", required=True, help="ZIP SLC input")
    pol_proc.add_argument("--output-dir", required=True, help="Directory output")

    wf = sub.add_parser(
        "run-parallel-workflow",
        help="Esegue workflow completo in parallelo: coherence(1-2,2-3), polarimetric(1,2), change polsar/coh",
    )
    wf.add_argument("--config", default="config.yaml", help="Config YAML/JSON")
    wf.add_argument("--selection-file", help="Percorso download_selection.json (opzionale)")
    wf.add_argument("--output-dir", default="workflow_parallel_outputs", help="Directory output workflow")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        return _cmd_download(Path(args.config))
    if args.command == "changedet-coh":
        return _cmd_changedet_coh(
            new_coh=Path(args.new_coh),
            old_coh=Path(args.old_coh),
            output_dir=Path(args.output_dir),
        )
    if args.command == "changedet-polsar":
        return _cmd_changedet_polsar(
            c2_a=Path(args.input_c2_a),
            c2_b=Path(args.input_c2_b),
            output_dir=Path(args.output_dir),
            boxcar=_parse_boxcar(args.boxcar),
        )
    if args.command == "process-coherence":
        return _cmd_process_coherence(
            config_path=Path(args.config),
            master_zip=Path(args.master_zip),
            slave_zip=Path(args.slave_zip),
            output_dir=Path(args.output_dir),
        )
    if args.command == "process-polarimetric":
        return _cmd_process_polarimetric(
            config_path=Path(args.config),
            input_zip=Path(args.input_zip),
            output_dir=Path(args.output_dir),
        )
    if args.command == "run-parallel-workflow":
        return _cmd_run_parallel_workflow(
            config_path=Path(args.config),
            selection_file=Path(args.selection_file) if args.selection_file else None,
            output_dir=Path(args.output_dir),
        )

    parser.error("Comando non supportato")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
