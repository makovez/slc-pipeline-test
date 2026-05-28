#!/usr/bin/env python3
"""Simple ASF downloader for Sentinel-1 SLC scenes.

Workflow:
1) Search and download 2 SLC products over a hardcoded bbox/date range.
2) Pick orbit/path metadata from one selected scene.
3) Search and download the closest scene about 1 month earlier with matching path/frame.

Authentication:
- Uses ASF_USERNAME / ASF_PASSWORD if available.
- Falls back to the provided Earthdata credentials.
- Ensure ASF EULA is accepted on your account, otherwise downloads return HTTP 401.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import asf_search as asf
from shapely.geometry import box as geom_box
from shapely.geometry import shape as geom_shape
from shapely.wkt import loads as load_wkt


def bbox_to_wkt(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    """Convert bbox to a WKT polygon for ASF intersectsWith."""
    return (
        "POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, "
        f"{min_lon} {max_lat}, "
        f"{min_lon} {min_lat}"
        "))"
    )


def parse_iso_utc(value: str) -> datetime:
    """Parse ASF ISO datetime safely to timezone-aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def scene_name(product: asf.ASFProduct) -> str:
    props = product.properties
    return (
        props.get("sceneName")
        or props.get("fileID")
        or props.get("granuleType")
        or "unknown_scene"
    )


def search_slc(
    intersects_wkt: str,
    start_dt: datetime,
    end_dt: datetime,
    max_results: int = 200,
) -> asf.ASFSearchResults:
    """Search Sentinel-1 SLC products in ASF."""
    return asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        intersectsWith=intersects_wkt,
        start=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        maxResults=max_results,
    )


def product_contains_bbox(
    product: asf.ASFProduct,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> bool:
    """Return True if ASF product footprint fully covers the bbox."""
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

    # `covers` includes touching boundaries; `contains` would be stricter.
    return footprint.covers(bbox_geom)


def filter_full_bbox_coverage(
    products: Iterable[asf.ASFProduct],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> List[asf.ASFProduct]:
    return [
        p
        for p in products
        if product_contains_bbox(p, min_lon, min_lat, max_lon, max_lat)
    ]


def unique_products(products: Iterable[asf.ASFProduct]) -> List[asf.ASFProduct]:
    seen = set()
    out: List[asf.ASFProduct] = []
    for prod in products:
        key = prod.properties.get("sceneName") or prod.properties.get("fileID") or prod.properties.get("url")
        if key and key not in seen:
            out.append(prod)
            seen.add(key)
    return out


def select_closest_by_date(products: Iterable[asf.ASFProduct], target_dt: datetime) -> Optional[asf.ASFProduct]:
    best = None
    best_delta = None
    for prod in products:
        dt = parse_iso_utc(prod.properties["startTime"])
        delta = abs((dt - target_dt).total_seconds())
        if best is None or delta < best_delta:
            best = prod
            best_delta = delta
    return best


def download_products(products: Iterable[asf.ASFProduct], output_dir: Path, session: asf.ASFSession) -> None:
    products = list(products)
    if not products:
        print("Nessun prodotto da scaricare.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = asf.ASFSearchResults(products)
    print(f"Inizio download di {len(products)} prodotto/i in: {output_dir}")
    results.download(path=str(output_dir), session=session, processes=1)


def main() -> None:
    # Hardcoded area: Emilia  bbox (lon/lat)
    min_lon, min_lat, max_lon, max_lat = (
        11.5891752243041992,
        44.3745611959093438,
        12.1745191687090415,
        44.6658172607421875,
    )

    # Hardcoded initial date range (UTC)
    range_start = datetime(2026, 3, 28, 0, 0, 0, tzinfo=timezone.utc)
    range_end = datetime(2026, 4, 6, 23, 59, 59, tzinfo=timezone.utc)

    output_dir = Path("Emilia_2")

    print("Autenticazione ASF...")
    username = os.getenv("ASF_USERNAME", "makovez")
    password = os.getenv("ASF_PASSWORD", "Akcak4XP4rgL?hG")
    session = asf.ASFSession().auth_with_creds(username, password)

    intersects = bbox_to_wkt(min_lon, min_lat, max_lon, max_lat)
    print("Ricerca iniziale SLC su bbox/date...")
    initial_results = search_slc(intersects, range_start, range_end, max_results=200)
    initial_results = filter_full_bbox_coverage(initial_results, min_lon, min_lat, max_lon, max_lat)
    print(f"Risultati che coprono interamente il bbox: {len(initial_results)}")

    if len(initial_results) == 0:
        print("Nessun prodotto SLC trovato nel range iniziale.")
        return

    # Sort newest first, then pick one reference product.
    initial_sorted = sorted(initial_results, key=lambda p: p.properties["startTime"], reverse=True)
    ref = initial_sorted[0]

    # Select orbit/path reference from the newest scene.
    ref_props = ref.properties
    ref_orbit = ref_props.get("orbit")
    ref_path = ref_props.get("pathNumber")
    ref_frame = ref_props.get("frameNumber")
    ref_dir = ref_props.get("flightDirection")
    ref_start = parse_iso_utc(ref_props["startTime"])

    print("\nRiferimento scelto:")
    print(
        f"  scena={scene_name(ref)}\n"
        f"  orbit={ref_orbit}, path={ref_path}, frame={ref_frame}, direction={ref_dir}, start={ref_start.isoformat()}"
    )

    # Target around one month before the reference acquisition.
    target_dt = ref_start - timedelta(days=30)
    window_start = target_dt - timedelta(days=20)
    window_end = target_dt + timedelta(days=20)

    print(
        "\nRicerca scena ~1 mese prima con stesso path/frame/direction "
        f"tra {window_start.isoformat()} e {window_end.isoformat()}"
    )
    month_results = search_slc(intersects, window_start, window_end, max_results=300)
    month_results = filter_full_bbox_coverage(month_results, min_lon, min_lat, max_lon, max_lat)
    print(f"Risultati mese precedente con copertura completa bbox: {len(month_results)}")

    matched = [
        p
        for p in month_results
        if p.properties.get("pathNumber") == ref_path
        and p.properties.get("frameNumber") == ref_frame
        and p.properties.get("flightDirection") == ref_dir
    ]

    month_before = select_closest_by_date(matched, target_dt)
    if month_before is None:
        print("Nessuna scena matchata per path/frame/direction a ~1 mese prima.")
        to_download = unique_products([ref])
    else:
        mb_props = month_before.properties
        print("Scena 1 mese prima selezionata:")
        print(
            f"  {scene_name(month_before)} | start={mb_props.get('startTime')} "
            f"| orbit={mb_props.get('orbit')} | path={mb_props.get('pathNumber')} "
            f"| frame={mb_props.get('frameNumber')} | dir={mb_props.get('flightDirection')}"
        )
        to_download = unique_products([ref, month_before])

    print(f"\nTotale prodotti da scaricare: {len(to_download)}")
    download_products(to_download, output_dir, session)


if __name__ == "__main__":
    main()
