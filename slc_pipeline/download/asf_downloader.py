from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import asf_search as asf
from shapely.geometry import box as geom_box
from shapely.geometry import shape as geom_shape
from shapely.wkt import loads as load_wkt

from slc_pipeline.config import PipelineConfig, ReferenceOffset


@dataclass(frozen=True)
class SceneSelection:
    event_scene_name: str
    event_start_utc: str
    reference_scene_names: list[str]
    output_dir: str


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _scene_name(product: asf.ASFProduct) -> str:
    props = product.properties
    return (
        props.get("sceneName")
        or props.get("fileID")
        or props.get("granuleType")
        or "unknown_scene"
    )


def _path_value(props: dict) -> str | None:
    value = props.get("pathNumber")
    if value is None:
        return None
    return str(value)


def _norm(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _unique_products(products: Iterable[asf.ASFProduct]) -> list[asf.ASFProduct]:
    seen = set()
    out: list[asf.ASFProduct] = []
    for prod in products:
        key = (
            prod.properties.get("sceneName")
            or prod.properties.get("fileID")
            or prod.properties.get("url")
        )
        if key and key not in seen:
            out.append(prod)
            seen.add(key)
    return out


def _select_closest_by_date(products: Iterable[asf.ASFProduct], target_dt: datetime) -> asf.ASFProduct | None:
    best = None
    best_delta = None
    for prod in products:
        dt = _parse_iso_utc(prod.properties["startTime"])
        delta = abs((dt - target_dt).total_seconds())
        if best is None or delta < best_delta:
            best = prod
            best_delta = delta
    return best


class ASFDownloader:
    def __init__(self, config: PipelineConfig):
        self.config = config
        username, password = config.asf.resolve_credentials()
        self.session = asf.ASFSession().auth_with_creds(username, password)

    def run_download_triplet(self) -> SceneSelection:
        event = self._select_event_scene()
        refs = self._select_reference_scenes(event)
        to_download = _unique_products([event] + refs)
        self._download_products(to_download)

        event_name = _scene_name(event)
        event_start = event.properties.get("startTime", "")
        ref_names = [_scene_name(p) for p in refs]

        return SceneSelection(
            event_scene_name=event_name,
            event_start_utc=event_start,
            reference_scene_names=ref_names,
            output_dir=str(self.config.download.output_dir),
        )

    def _search_slc(
        self,
        start_dt: datetime,
        end_dt: datetime,
        max_results: int,
    ) -> asf.ASFSearchResults:
        return asf.geo_search(
            platform=[asf.PLATFORM.SENTINEL1],
            processingLevel=[asf.PRODUCT_TYPE.SLC],
            intersectsWith=self.config.bbox.to_wkt(),
            start=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            maxResults=max_results,
        )

    def _select_event_scene(self) -> asf.ASFProduct:
        start_dt = self.config.event_date_range.start
        end_dt = self.config.event_date_range.end
        print(f"Ricerca scena flood tra {start_dt.isoformat()} e {end_dt.isoformat()}...")

        results = self._search_slc(start_dt, end_dt, self.config.download.max_results_event)
        full_cov = self._filter_full_bbox_coverage(results)
        if not full_cov:
            raise RuntimeError("Nessuna scena flood trovata con copertura completa del bbox")

        event = sorted(full_cov, key=lambda p: p.properties["startTime"], reverse=True)[0]
        props = event.properties
        print(
            "Flood scene selezionata:\n"
            f"  scena={_scene_name(event)}\n"
            f"  orbit={props.get('orbit')}, path={props.get('pathNumber')}, frame={props.get('frameNumber')}, "
            f"direction={props.get('flightDirection')}, start={props.get('startTime')}"
        )
        return event

    def _select_reference_scenes(self, event: asf.ASFProduct) -> list[asf.ASFProduct]:
        ref_props = event.properties
        ref_path = _path_value(ref_props)
        ref_dir = _norm(ref_props.get("flightDirection"))
        ref_start = _parse_iso_utc(ref_props["startTime"])

        print(
            "Filtro riferimenti su: "
            f"ref_path={ref_path} (pathNumber della flood scene), "
            f"ref_direction={ref_dir}"
        )

        selected: list[asf.ASFProduct] = []
        for idx, rule in enumerate(self.config.references[:2]):
            older_than_start: datetime | None = None
            if idx == 1 and selected:
                older_than_start = _parse_iso_utc(selected[0].properties["startTime"])

            product = self._select_one_reference(
                rule=rule,
                ref_start=ref_start,
                ref_path=ref_path,
                ref_dir=ref_dir,
                already_selected=selected,
                older_than_start=older_than_start,
            )
            if product is not None:
                selected.append(product)

        if len(selected) < 2:
            raise RuntimeError(
                "Riferimenti insufficienti: trovate meno di 2 scene storiche con i vincoli richiesti"
            )

        # Keep references ordered by time: second product newer, third product older.
        selected = sorted(
            selected,
            key=lambda p: _parse_iso_utc(p.properties["startTime"]),
            reverse=True,
        )
        if _parse_iso_utc(selected[1].properties["startTime"]) >= _parse_iso_utc(
            selected[0].properties["startTime"]
        ):
            raise RuntimeError("Vincolo non soddisfatto: il terzo prodotto deve essere piu vecchio del secondo")

        return selected

    def _select_one_reference(
        self,
        rule: ReferenceOffset,
        ref_start: datetime,
        ref_path: object,
        ref_dir: object,
        already_selected: list[asf.ASFProduct],
        older_than_start: datetime | None,
    ) -> asf.ASFProduct | None:
        target_dt = ref_start - timedelta(days=rule.offset_days)
        window_start = target_dt - timedelta(days=rule.search_window_days)
        window_end = target_dt + timedelta(days=rule.search_window_days)

        print(
            f"Ricerca riferimento offset={rule.offset_days}gg "
            f"in finestra [{window_start.isoformat()} .. {window_end.isoformat()}]"
        )
        results = self._search_slc(
            window_start,
            window_end,
            self.config.download.max_results_reference,
        )
        full_cov = self._filter_full_bbox_coverage(results)

        used_names = {_scene_name(p) for p in already_selected}
        matched = []
        path_pass = 0
        dir_pass = 0
        time_pass = 0
        for p in full_cov:
            props = p.properties
            if _scene_name(p) in used_names:
                continue
            cand_path = _path_value(props)
            if cand_path != _norm(ref_path):
                continue
            path_pass += 1

            cand_dir = _norm(props.get("flightDirection"))
            if cand_dir != ref_dir:
                continue
            dir_pass += 1

            if older_than_start is not None:
                p_start = _parse_iso_utc(props["startTime"])
                if p_start >= older_than_start:
                    continue
            time_pass += 1
            matched.append(p)

        print(
            "  Diagnostica filtri: "
            f"bbox={len(full_cov)}, path_ok={path_pass}, dir_ok={dir_pass}, time_ok={time_pass}, matched={len(matched)}"
        )

        picked = _select_closest_by_date(matched, target_dt)
        if picked is None:
            print("  Nessuna scena compatibile per questo offset")
            return None

        props = picked.properties
        print(
            "  Riferimento selezionato: "
            f"{_scene_name(picked)} | start={props.get('startTime')} | path={_path_value(props)} "
            f"| dir={props.get('flightDirection')}"
        )
        return picked

    def _download_products(self, products: Iterable[asf.ASFProduct]) -> None:
        products = list(products)
        if not products:
            raise RuntimeError("Nessun prodotto da scaricare")

        output_dir = self.config.download.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        processes = min(len(products), max(1, self.config.download.parallel_processes))

        print(
            f"Download di {len(products)} prodotto/i in {output_dir} "
            f"con {processes} processi paralleli..."
        )
        asf.ASFSearchResults(products).download(
            path=str(output_dir),
            session=self.session,
            processes=processes,
        )

    def _filter_full_bbox_coverage(self, products: Iterable[asf.ASFProduct]) -> list[asf.ASFProduct]:
        min_lon = self.config.bbox.min_lon
        min_lat = self.config.bbox.min_lat
        max_lon = self.config.bbox.max_lon
        max_lat = self.config.bbox.max_lat

        return [
            p
            for p in products
            if _product_contains_bbox(p, min_lon, min_lat, max_lon, max_lat)
        ]


def _product_contains_bbox(
    product: asf.ASFProduct,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> bool:
    bbox_geom = geom_box(min_lon, min_lat, max_lon, max_lat)

    geometry = getattr(product, "geometry", None)

    if not geometry:
        geojson_fn = getattr(product, "geojson", None)
        if callable(geojson_fn):
            try:
                geojson = geojson_fn()
                if isinstance(geojson, dict):
                    geometry = geojson.get("geometry")
            except Exception:
                geometry = None

    if not geometry:
        geometry = product.properties.get("geometry")

    if not geometry:
        return False

    try:
        if isinstance(geometry, dict):
            footprint = geom_shape(geometry)
        elif isinstance(geometry, str):
            footprint = load_wkt(geometry)
        else:
            return False
    except Exception:
        return False

    return footprint.covers(bbox_geom)
