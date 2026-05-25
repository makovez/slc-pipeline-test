#!/usr/bin/env python3
"""Generic Sentinel-1 SLC processing pipeline using SNAP GPT.

This script processes two SLC products from an input directory and produces:
1) GRD-like sigma0 (dB) GeoTIFF for each date
2) Polarimetric coherence GeoTIFF for each date (from C2 matrix)
3) Interferometric coherence GeoTIFF for the date pair

The area of interest is controlled by a geographic bbox and is applied automatically.
No hardcoded pixel subset is used.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import MultiPoint, box, mapping


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_wkt(self) -> str:
        return (
            "POLYGON(("
            f"{self.min_lon} {self.min_lat}, "
            f"{self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, "
            f"{self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}"
            "))"
        )


def xml_value(parent: ET.Element, tag: str, text: str) -> ET.Element:
    elem = ET.SubElement(parent, tag)
    elem.text = text
    return elem


def graph_root() -> ET.Element:
    root = ET.Element("graph", {"id": "Graph"})
    version = ET.SubElement(root, "version")
    version.text = "1.0"
    return root


def node(graph: ET.Element, node_id: str, operator: str, source_ref: str | None = None) -> ET.Element:
    n = ET.SubElement(graph, "node", {"id": node_id})
    op = ET.SubElement(n, "operator")
    op.text = operator
    sources = ET.SubElement(n, "sources")
    if source_ref:
        ET.SubElement(sources, "sourceProduct", {"refid": source_ref})
    params = ET.SubElement(n, "parameters", {"class": "com.bc.ceres.binding.dom.XppDomElement"})
    return params


def write_graph(graph: ET.Element, out_xml: Path) -> None:
    tree = ET.ElementTree(graph)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def run_gpt(gpt_path: Path, xml_path: Path, extra_args: Sequence[str] | None = None) -> None:
    cmd = [str(gpt_path), str(xml_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Eseguo: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GPT failed with exit code {proc.returncode}\n{proc.stdout}")


def detect_gpt_path() -> Path:
    env_path = os.environ.get("SNAP_GPT_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"SNAP_GPT_PATH impostato ma non valido: {p}")

    candidates = [
        Path("/Applications/esa-snap/bin/gpt"),
        Path("/usr/local/snap/bin/gpt"),
        Path("/opt/snap/bin/gpt"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    which = shutil.which("gpt")
    if which:
        p = Path(which)
        # On macOS /usr/sbin/gpt is disk partition tool, not SNAP.
        if "esa-snap" in str(p) or "snap/bin" in str(p):
            return p

    raise FileNotFoundError(
        "SNAP GPT non trovato. Imposta SNAP_GPT_PATH=/Applications/esa-snap/bin/gpt"
    )


def find_slc_archives(input_dir: Path) -> List[Path]:
    products = sorted(input_dir.glob("*.zip"))
    if len(products) < 2:
        raise FileNotFoundError(
            f"Servono almeno 2 prodotti SLC in {input_dir}, trovati {len(products)}"
        )
    return products


def scene_start_token(path: Path) -> str:
    m = re.search(r"_(\d{8}T\d{6})_", path.name)
    if not m:
        raise ValueError(f"Impossibile estrarre data scena dal filename: {path.name}")
    return m.group(1)


def identify_scenes(products: Iterable[Path]) -> List[Path]:
    scenes = list(products)
    scenes.sort(key=lambda p: scene_start_token(p))
    return scenes


def bbox_centered_pixel_window(base: BBox, pixel_spacing_m: float, width_px: int, height_px: int) -> BBox:
    # Approximate meter-to-degree conversion around AOI center latitude.
    center_lon = (base.min_lon + base.max_lon) / 2.0
    center_lat = (base.min_lat + base.max_lat) / 2.0

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * max(math.cos(math.radians(center_lat)), 1e-6)

    half_width_m = (width_px * pixel_spacing_m) / 2.0
    half_height_m = (height_px * pixel_spacing_m) / 2.0

    half_lon_deg = half_width_m / meters_per_deg_lon
    half_lat_deg = half_height_m / meters_per_deg_lat

    return BBox(
        min_lon=center_lon - half_lon_deg,
        min_lat=center_lat - half_lat_deg,
        max_lon=center_lon + half_lon_deg,
        max_lat=center_lat + half_lat_deg,
    )


def _strip_xml_namespaces(root: ET.Element) -> None:
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _annotation_member_for_subswath(
    zip_path: Path,
    subswath: str,
    pol_priority: Sequence[str],
) -> str:
    sw = subswath.lower()
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for m in zf.namelist():
            ml = m.lower()
            if not ml.endswith(".xml"):
                continue
            if "/annotation/" not in ml:
                continue
            # Use only main measurement annotation XML, not calibration/rfi XML.
            if "/annotation/calibration/" in ml or "/annotation/rfi/" in ml:
                continue
            members.append(m)
        for pol in pol_priority:
            token = f"-{sw}-slc-{pol.lower()}-"
            hits = [m for m in members if token in m.lower()]
            if hits:
                return sorted(hits)[0]
    raise ValueError(f"Annotation XML non trovato per subswath {subswath} in {zip_path}")


def overlapping_burst_range_for_bbox(
    input_zip: Path,
    subswath: str,
    bbox: BBox,
    pol_priority: Sequence[str] = ("vv", "vh"),
) -> Tuple[int, int]:
    """Return 1-based burst range overlapping bbox for TOPSAR-Split.

    Uses geolocation grid points from annotation XML and maps each point to
    a burst using linesPerBurst.
    """
    annotation_member = _annotation_member_for_subswath(input_zip, subswath, pol_priority)
    with zipfile.ZipFile(input_zip) as zf:
        with zf.open(annotation_member) as fh:
            root = ET.fromstring(fh.read())

    _strip_xml_namespaces(root)

    lines_per_burst_text = root.findtext(".//swathTiming/linesPerBurst")
    if not lines_per_burst_text:
        raise ValueError("linesPerBurst non trovato nell'annotation XML")
    lines_per_burst = int(lines_per_burst_text)

    burst_nodes = root.findall(".//swathTiming/burstList/burst")
    burst_count = len(burst_nodes)
    if burst_count == 0:
        raise ValueError("burstList vuota nell'annotation XML")

    aoi = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    points_by_line: dict[int, list[tuple[float, float]]] = {}

    for gp in root.findall(".//geolocationGrid/geolocationGridPointList/geolocationGridPoint"):
        line_txt = gp.findtext("line")
        lat_txt = gp.findtext("latitude")
        lon_txt = gp.findtext("longitude")
        if not line_txt or not lat_txt or not lon_txt:
            continue

        line = int(line_txt)
        lat = float(lat_txt)
        lon = float(lon_txt)
        points_by_line.setdefault(line, []).append((lon, lat))

    overlapping: list[int] = []

    unique_lines = sorted(points_by_line.keys())
    if len(unique_lines) >= burst_count + 1:
        # Sentinel-1 annotation usually stores geolocation points at burst boundaries.
        # Build each burst footprint from two consecutive boundary lines.
        for idx in range(1, burst_count + 1):
            top_line = unique_lines[idx - 1]
            bottom_line = unique_lines[idx]
            pts = points_by_line.get(top_line, []) + points_by_line.get(bottom_line, [])
            if len(pts) < 3:
                continue
            footprint = MultiPoint(pts).convex_hull
            if footprint.is_empty:
                continue
            if footprint.intersects(aoi):
                overlapping.append(idx)
    else:
        # Fallback for non-standard geolocation grids: map points to bursts by line index.
        per_burst_points: dict[int, list[tuple[float, float]]] = {i: [] for i in range(1, burst_count + 1)}
        for line, pts in points_by_line.items():
            burst_idx = (line // lines_per_burst) + 1
            if burst_idx < 1:
                burst_idx = 1
            if burst_idx > burst_count:
                burst_idx = burst_count
            per_burst_points[burst_idx].extend(pts)

        for idx in range(1, burst_count + 1):
            pts = per_burst_points[idx]
            if len(pts) < 3:
                continue
            footprint = MultiPoint(pts).convex_hull
            if footprint.is_empty:
                continue
            if footprint.intersects(aoi):
                overlapping.append(idx)

    if not overlapping:
        raise ValueError(f"Nessun burst di {subswath} interseca il bbox richiesto")

    return min(overlapping), max(overlapping)


def clip_geotiff_to_bbox(src_tif: Path, dst_tif: Path, bbox: BBox, nodata: float | None = None) -> None:
    geom = [mapping(box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat))]
    with rasterio.open(src_tif) as src:
        out_image, out_transform = mask(src, geom, crop=True, nodata=nodata)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        if nodata is not None:
            out_meta["nodata"] = nodata

    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, "w", **out_meta) as dst:
        dst.write(out_image)


def raster_has_valid_values(path: Path, min_valid_ratio: float = 0.01) -> bool:
    """Return True when raster contains enough finite pixels."""
    with rasterio.open(path) as src:
        arr = src.read(1)

    finite = np.isfinite(arr)
    if not finite.any():
        return False

    ratio = float(finite.sum()) / float(arr.size)
    return ratio >= min_valid_ratio


def linear_to_db(src_tif: Path, dst_tif: Path, eps: float = 1e-10) -> None:
    with rasterio.open(src_tif) as src:
        arr = src.read(1).astype(np.float32)
        valid = np.isfinite(arr) & (arr > 0)
        db = np.full(arr.shape, np.nan, dtype=np.float32)
        db[valid] = 10.0 * np.log10(np.maximum(arr[valid], eps))
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, compress="deflate", nodata=np.nan)

    with rasterio.open(dst_tif, "w", **meta) as dst:
        dst.write(db, 1)


def infer_c2_band_indices(descriptions: Sequence[str | None]) -> Tuple[int, int, int, int]:
    names = [((d or "").strip().lower()) for d in descriptions]

    def idx_contains(tokens: Sequence[str]) -> int | None:
        for i, n in enumerate(names):
            if all(t in n for t in tokens):
                return i
        return None

    c11 = idx_contains(["c11"])
    c22 = idx_contains(["c22"])
    c12_real = idx_contains(["c12", "real"])
    c12_imag = idx_contains(["c12", "imag"])

    # Fallback to first 4 bands if descriptions are missing.
    if None in (c11, c12_real, c12_imag, c22):
        if len(descriptions) < 4:
            raise ValueError("Output C2 inatteso: servono almeno 4 bande")
        return 0, 1, 2, 3
    return c11, c12_real, c12_imag, c22



def clip_c2_bands(
    src_tif: Path,
    dst_tif: Path,
    upper_percentile_diag: float = 99.9,
    upper_percentile_cross_abs: float = 99.9,
) -> None:
    """Clip C2 bands to reduce extreme outliers before coherence computation.

    C11/C22 are constrained to [0, pX], while C12 real/imag are constrained
    symmetrically to [-pX_abs, +pX_abs].
    """
    with rasterio.open(src_tif) as src:
        descriptions = src.descriptions
        c11_i, c12r_i, c12i_i, c22_i = infer_c2_band_indices(descriptions)

        clipped = src.read().astype(np.float32)
        band_map = {
            c11_i: "c11",
            c12r_i: "c12_real",
            c12i_i: "c12_imag",
            c22_i: "c22",
        }

        for idx, name in band_map.items():
            arr = clipped[idx]
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                continue

            if name in ("c11", "c22"):
                high = np.percentile(valid, upper_percentile_diag)
                clipped[idx] = np.clip(arr, 0.0, high)
                print(f"Clip {name}: [0, {high:.6g}] (p{upper_percentile_diag})")
            else:
                high_abs = np.percentile(np.abs(valid), upper_percentile_cross_abs)
                clipped[idx] = np.clip(arr, -high_abs, high_abs)
                print(
                    f"Clip {name}: [{-high_abs:.6g}, {high_abs:.6g}] "
                    f"(abs p{upper_percentile_cross_abs})"
                )

        meta = src.meta.copy()
        meta.update(dtype="float32", compress="deflate")

    with rasterio.open(dst_tif, "w", **meta) as dst:
        dst.write(clipped)


def build_grd_graph(
    input_zip: Path,
    out_tif: Path,
    bbox: BBox,
    dem_name: str,
    pixel_spacing_m: float,
    subswath: str,
) -> ET.Element:
    g = graph_root()
    first_burst, last_burst = overlapping_burst_range_for_bbox(input_zip, subswath, bbox)

    p = node(g, "Read", "Read")
    xml_value(p, "file", str(input_zip))

    p = node(g, "TOPSAR-Split", "TOPSAR-Split", source_ref="Read")
    xml_value(p, "subswath", subswath)
    xml_value(p, "selectedPolarisations", "VH,VV")
    xml_value(p, "firstBurstIndex", str(first_burst))
    xml_value(p, "lastBurstIndex", str(last_burst))
    xml_value(p, "wktAoi", bbox.to_wkt())

    p = node(g, "Apply-Orbit-File", "Apply-Orbit-File", source_ref="TOPSAR-Split")
    xml_value(p, "orbitType", "Sentinel Precise (Auto Download)")
    xml_value(p, "polyDegree", "3")
    xml_value(p, "continueOnFail", "true")

    p = node(g, "Calibration", "Calibration", source_ref="Apply-Orbit-File")
    xml_value(p, "outputImageInComplex", "false")
    xml_value(p, "outputSigmaBand", "true")
    xml_value(p, "createGammaBand", "false")
    xml_value(p, "createBetaBand", "false")

    p = node(g, "TOPSAR-Deburst", "TOPSAR-Deburst", source_ref="Calibration")
    xml_value(p, "selectedPolarisations", "VH,VV")

    p = node(g, "Multilook", "Multilook", source_ref="TOPSAR-Deburst")
    xml_value(p, "nRgLooks", "4")
    xml_value(p, "nAzLooks", "1")
    xml_value(p, "outputIntensity", "true")
    xml_value(p, "grSquarePixel", "true")

    # Terrain-Flattening is not valid for sigma0-only GRD branch.
    p = node(g, "Terrain-Correction", "Terrain-Correction", source_ref="Multilook")
    xml_value(p, "demName", dem_name)
    xml_value(p, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "imgResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "pixelSpacingInMeter", str(pixel_spacing_m))
    xml_value(p, "saveSelectedSourceBand", "true")
    xml_value(p, "outputComplex", "false")

    p = node(g, "Write", "Write", source_ref="Terrain-Correction")
    xml_value(p, "file", str(out_tif))
    xml_value(p, "formatName", "GeoTIFF")

    return g


def build_polarimetric_graph(
    input_zip: Path,
    out_tif: Path,
    bbox: BBox,
    subswath: str,
) -> ET.Element:
    g = graph_root()
    first_burst, last_burst = overlapping_burst_range_for_bbox(input_zip, subswath, bbox)

    p = node(g, "Read", "Read")
    xml_value(p, "useAdvancedOptions", "false")
    xml_value(p, "file", str(input_zip))
    xml_value(p, "copyMetadata", "true")
    xml_value(p, "bandNames", "")
    xml_value(p, "maskNames", "")

    p = node(g, "TOPSAR-Split", "TOPSAR-Split", source_ref="Read")
    xml_value(p, "subswath", subswath)
    xml_value(p, "selectedPolarisations", "VH,VV")
    xml_value(p, "firstBurstIndex", str(first_burst))
    xml_value(p, "lastBurstIndex", str(last_burst))
    xml_value(p, "wktAoi", "")

    p = node(g, "Apply-Orbit-File", "Apply-Orbit-File", source_ref="TOPSAR-Split")
    xml_value(p, "orbitType", "Sentinel Precise (Auto Download)")
    xml_value(p, "polyDegree", "3")
    xml_value(p, "continueOnFail", "true")

    p = node(g, "Calibration", "Calibration", source_ref="Apply-Orbit-File")
    xml_value(p, "sourceBands", "")
    xml_value(p, "auxFile", "Latest Auxiliary File")
    xml_value(p, "externalAuxFile", "")
    xml_value(p, "outputImageInComplex", "true")
    xml_value(p, "outputImageScaleInDb", "false")
    xml_value(p, "createGammaBand", "false")
    xml_value(p, "createBetaBand", "false")
    xml_value(p, "selectedPolarisations", "")
    xml_value(p, "outputSigmaBand", "true")
    xml_value(p, "outputGammaBand", "false")
    xml_value(p, "outputBetaBand", "false")

    p = node(g, "TOPSAR-Deburst", "TOPSAR-Deburst", source_ref="Calibration")
    xml_value(p, "selectedPolarisations", "VH,VV")

    p = node(g, "Polarimetric-Matrices", "Polarimetric-Matrices", source_ref="TOPSAR-Deburst")
    xml_value(p, "matrix", "C2")

    p = node(g, "Multilook", "Multilook", source_ref="Polarimetric-Matrices")
    xml_value(p, "sourceBands", "")
    xml_value(p, "nRgLooks", "4")
    xml_value(p, "nAzLooks", "1")
    xml_value(p, "outputIntensity", "false")
    xml_value(p, "grSquarePixel", "true")

    p = node(g, "Terrain-Flattening", "Terrain-Flattening", source_ref="Multilook")
    xml_value(p, "sourceBands", "")
    xml_value(p, "demName", "SRTM 1Sec HGT")
    xml_value(p, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "externalDEMFile", "")
    xml_value(p, "externalDEMNoDataValue", "0.0")
    xml_value(p, "externalDEMApplyEGM", "false")
    xml_value(p, "outputSimulatedImage", "false")
    xml_value(p, "outputSigma0", "false")
    xml_value(p, "nodataValueAtSea", "false")
    xml_value(p, "additionalOverlap", "0.1")
    xml_value(p, "oversamplingMultiple", "1.0")

    p = node(g, "Terrain-Correction", "Terrain-Correction", source_ref="Terrain-Flattening")
    xml_value(p, "sourceBands", "")
    xml_value(p, "demName", "SRTM 1Sec HGT")
    xml_value(p, "externalDEMFile", "")
    xml_value(p, "externalDEMNoDataValue", "0.0")
    xml_value(p, "externalDEMApplyEGM", "true")
    xml_value(p, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "imgResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "pixelSpacingInMeter", "13.92")
    xml_value(p, "pixelSpacingInDegree", "1.2504548754943738E-4")
    xml_value(
        p,
        "mapProjection",
        'GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH]]',
    )
    xml_value(p, "alignToStandardGrid", "false")
    xml_value(p, "standardGridOriginX", "0.0")
    xml_value(p, "standardGridOriginY", "0.0")
    xml_value(p, "nodataValueAtSea", "false")
    xml_value(p, "saveDEM", "false")
    xml_value(p, "saveLatLon", "false")
    xml_value(p, "saveIncidenceAngleFromEllipsoid", "false")
    xml_value(p, "saveLocalIncidenceAngle", "false")
    xml_value(p, "saveProjectedLocalIncidenceAngle", "false")
    xml_value(p, "saveSelectedSourceBand", "true")
    xml_value(p, "saveLayoverShadowMask", "false")
    xml_value(p, "outputComplex", "false")
    xml_value(p, "applyRadiometricNormalization", "false")
    xml_value(p, "saveSigmaNought", "false")
    xml_value(p, "saveGammaNought", "false")
    xml_value(p, "saveBetaNought", "false")
    xml_value(p, "incidenceAngleForSigma0", "Use projected local incidence angle from DEM")
    xml_value(p, "incidenceAngleForGamma0", "Use projected local incidence angle from DEM")
    xml_value(p, "auxFile", "Latest Auxiliary File")
    xml_value(p, "externalAuxFile", "")

    p = node(g, "Subset", "Subset", source_ref="Terrain-Correction")
    xml_value(p, "sourceBands", "")
    xml_value(p, "tiePointGrids", "")
    xml_value(p, "region", "")
    xml_value(p, "referenceBand", "")
    xml_value(p, "geoRegion", bbox.to_wkt())
    xml_value(p, "subSamplingX", "1")
    xml_value(p, "subSamplingY", "1")
    xml_value(p, "fullSwath", "false")
    xml_value(p, "vectorFile", "")
    xml_value(p, "polygonRegion", "")
    xml_value(p, "copyMetadata", "true")

    p = node(g, "Write", "Write", source_ref="Subset")
    xml_value(p, "file", str(out_tif))
    xml_value(p, "formatName", "GeoTIFF")

    return g


def build_interferometric_coherence_graph(
    master_zip: Path,
    slave_zip: Path,
    out_tif: Path,
    bbox: BBox,
    dem_name: str,
    pixel_spacing_m: float,
    subswath: str,
) -> ET.Element:
    g = graph_root()
    master_first, master_last = overlapping_burst_range_for_bbox(master_zip, subswath, bbox, pol_priority=("vv",))
    slave_first, slave_last = overlapping_burst_range_for_bbox(slave_zip, subswath, bbox, pol_priority=("vv",))

    print(master_first, master_last)
    p = node(g, "Read_Master", "Read")
    xml_value(p, "file", str(master_zip))

    p = node(g, "Read_Slave", "Read")
    xml_value(p, "file", str(slave_zip))

    p = node(g, "TOPSAR-Split_Master", "TOPSAR-Split", source_ref="Read_Master")
    xml_value(p, "subswath", subswath)
    xml_value(p, "selectedPolarisations", "VV")
    xml_value(p, "firstBurstIndex", str(master_first))
    xml_value(p, "lastBurstIndex", str(master_last))
    xml_value(p, "wktAoi", "")

    p = node(g, "TOPSAR-Split_Slave", "TOPSAR-Split", source_ref="Read_Slave")
    xml_value(p, "subswath", subswath)
    xml_value(p, "selectedPolarisations", "VV")
    xml_value(p, "firstBurstIndex", str(slave_first))
    xml_value(p, "lastBurstIndex", str(slave_last))
    xml_value(p, "wktAoi", "")

    p = node(g, "Apply-Orbit_Master", "Apply-Orbit-File", source_ref="TOPSAR-Split_Master")
    xml_value(p, "orbitType", "Sentinel Precise (Auto Download)")
    xml_value(p, "continueOnFail", "true")

    p = node(g, "Apply-Orbit_Slave", "Apply-Orbit-File", source_ref="TOPSAR-Split_Slave")
    xml_value(p, "orbitType", "Sentinel Precise (Auto Download)")
    xml_value(p, "continueOnFail", "true")

    # Build stack by giving slave as second source in Back-Geocoding.
    n = ET.SubElement(g, "node", {"id": "Back-Geocoding"})
    op = ET.SubElement(n, "operator")
    op.text = "Back-Geocoding"
    sources = ET.SubElement(n, "sources")
    ET.SubElement(sources, "masterProduct", {"refid": "Apply-Orbit_Master"})
    ET.SubElement(sources, "slaveProduct", {"refid": "Apply-Orbit_Slave"})
    params = ET.SubElement(n, "parameters", {"class": "com.bc.ceres.binding.dom.XppDomElement"})
    xml_value(params, "demName", dem_name)
    xml_value(params, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(params, "externalDEMFile", "")
    xml_value(params, "externalDEMNoDataValue", "0.0")
    xml_value(params, "resamplingType", "BILINEAR_INTERPOLATION")
    xml_value(params, "maskOutAreaWithoutElevation", "true")
    xml_value(params, "outputRangeAzimuthOffset", "false")
    xml_value(params, "outputDerampDemodPhase", "false")
    xml_value(params, "disableReramp", "false")

    p = node(g, "Coherence", "Coherence", source_ref="Back-Geocoding")
    xml_value(p, "cohWinAz", "5")
    xml_value(p, "cohWinRg", "20")
    xml_value(p, "subtractFlatEarthPhase", "false")
    xml_value(p, "srpPolynomialDegree", "5")
    xml_value(p, "srpNumberPoints", "501")
    xml_value(p, "orbitDegree", "3")
    xml_value(p, "squarePixel", "true")
    xml_value(p, "subtractTopographicPhase", "false")
    xml_value(p, "demName", "SRTM 3Sec")
    xml_value(p, "externalDEMFile", "")
    xml_value(p, "externalDEMNoDataValue", "0.0")
    xml_value(p, "externalDEMApplyEGM", "true")
    xml_value(p, "tileExtensionPercent", "100")
    xml_value(p, "singleMaster", "true")

    p = node(g, "TOPSAR-Deburst", "TOPSAR-Deburst", source_ref="Coherence")
    xml_value(p, "selectedPolarisations", "VV")

    p = node(g, "Terrain-Correction", "Terrain-Correction", source_ref="TOPSAR-Deburst")
    xml_value(p, "sourceBands", "")
    xml_value(p, "demName", "SRTM 1Sec HGT")
    xml_value(p, "externalDEMFile", "")
    xml_value(p, "externalDEMNoDataValue", "0.0")
    xml_value(p, "externalDEMApplyEGM", "true")
    xml_value(p, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "imgResamplingMethod", "BILINEAR_INTERPOLATION")
    xml_value(p, "pixelSpacingInMeter", "13.92")
    xml_value(p, "pixelSpacingInDegree", "1.2504548754943738E-4")
    xml_value(
        p,
        "mapProjection",
        'GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH]]',
    )
    xml_value(p, "alignToStandardGrid", "false")
    xml_value(p, "standardGridOriginX", "0.0")
    xml_value(p, "standardGridOriginY", "0.0")
    xml_value(p, "nodataValueAtSea", "false")
    xml_value(p, "saveDEM", "false")
    xml_value(p, "saveLatLon", "false")
    xml_value(p, "saveIncidenceAngleFromEllipsoid", "false")
    xml_value(p, "saveLocalIncidenceAngle", "false")
    xml_value(p, "saveProjectedLocalIncidenceAngle", "false")
    xml_value(p, "saveSelectedSourceBand", "true")
    xml_value(p, "saveLayoverShadowMask", "false")
    xml_value(p, "outputComplex", "false")
    xml_value(p, "applyRadiometricNormalization", "false")
    xml_value(p, "saveSigmaNought", "false")
    xml_value(p, "saveGammaNought", "false")
    xml_value(p, "saveBetaNought", "false")
    xml_value(p, "incidenceAngleForSigma0", "Use projected local incidence angle from DEM")
    xml_value(p, "incidenceAngleForGamma0", "Use projected local incidence angle from DEM")
    xml_value(p, "auxFile", "Latest Auxiliary File")
    xml_value(p, "externalAuxFile", "")

    p = node(g, "Subset", "Subset", source_ref="Terrain-Correction")
    xml_value(p, "sourceBands", "")
    xml_value(p, "tiePointGrids", "")
    xml_value(p, "region", "")
    xml_value(p, "referenceBand", "")
    xml_value(p, "geoRegion", bbox.to_wkt())
    xml_value(p, "subSamplingX", "1")
    xml_value(p, "subSamplingY", "1")
    xml_value(p, "fullSwath", "false")
    xml_value(p, "vectorFile", "")
    xml_value(p, "polygonRegion", "")
    xml_value(p, "copyMetadata", "true")

    p = node(g, "Write", "Write", source_ref="Subset")
    xml_value(p, "file", str(out_tif))
    xml_value(p, "formatName", "GeoTIFF")

    return g


def scene_date_label(scene_zip: Path) -> str:
    return scene_start_token(scene_zip)[:8]


def save_run_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_coverage_in_bbox(
    raster_files: dict[str, Path],
    bbox: BBox,
    min_finite_pct_warning: float = 20.0,
) -> None:
    """Report pixel coverage within bbox for each raster.

    Prints:
    - finite coverage (% pixel not NaN/Inf)
    - non-zero coverage (% finite pixel with value != 0)
    """
    print("\n" + "="*70)
    print("ANALISI COPERTURA PIXEL NEL BBOX")
    print("="*70)
    
    geom = [mapping(box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat))]
    
    for label, raster_path in raster_files.items():
        if not raster_path.exists():
            print(f"\n{label}: FILE NON TROVATO ({raster_path})")
            continue
        
        try:
            with rasterio.open(raster_path) as src:
                # Clippa il raster al bbox
                clipped_data, _ = mask(src, geom, crop=True, nodata=np.nan)
                
                if clipped_data.ndim == 3:
                    # Multi-band: calcola per ogni banda
                    for band_idx in range(clipped_data.shape[0]):
                        arr = clipped_data[band_idx]
                        finite = np.isfinite(arr)
                        non_zero = finite & (arr != 0)
                        total = arr.size
                        finite_count = int(np.sum(finite))
                        non_zero_count = int(np.sum(non_zero))
                        finite_pct = (finite_count / total * 100) if total > 0 else 0
                        non_zero_pct = (non_zero_count / total * 100) if total > 0 else 0
                        print(f"\n{label} (banda {band_idx + 1}):")
                        print(f"  Pixel totali nel bbox: {total:,}")
                        print(f"  Pixel finiti (≠NaN, ≠Inf): {finite_count:,} ({finite_pct:.2f}%)")
                        print(f"  Pixel non-zero: {non_zero_count:,} ({non_zero_pct:.2f}%)")
                        if finite_pct < min_finite_pct_warning:
                            print(
                                f"  WARNING: copertura finita bassa (< {min_finite_pct_warning:.1f}%). "
                                "Probabile AOI fuori scena o bordi/no-data dominanti."
                            )
                else:
                    # Single-band
                    arr = clipped_data[0] if clipped_data.ndim == 3 else clipped_data
                    finite = np.isfinite(arr)
                    non_zero = finite & (arr != 0)
                    total = arr.size
                    finite_count = int(np.sum(finite))
                    non_zero_count = int(np.sum(non_zero))
                    finite_pct = (finite_count / total * 100) if total > 0 else 0
                    non_zero_pct = (non_zero_count / total * 100) if total > 0 else 0
                    print(f"\n{label}:")
                    print(f"  Pixel totali nel bbox: {total:,}")
                    print(f"  Pixel finiti (≠NaN, ≠Inf): {finite_count:,} ({finite_pct:.2f}%)")
                    print(f"  Pixel non-zero: {non_zero_count:,} ({non_zero_pct:.2f}%)")
                    if finite_pct < min_finite_pct_warning:
                        print(
                            f"  WARNING: copertura finita bassa (< {min_finite_pct_warning:.1f}%). "
                            "Probabile AOI fuori scena o bordi/no-data dominanti."
                        )
        except Exception as e:
            print(f"\n{label}: ERRORE - {e}")
    
    print("="*70)


def main() -> None:
    # Hardcoded runtime configuration as requested.
    bbox = BBox(
        min_lon=11.5891752243041992,
        min_lat=44.3745611959093438,
        max_lon=12.1745191687090415,
        max_lat=44.6658172607421875,
    )
    target_width_px = 512
    target_height_px = 512
    input_dir = Path("frame1_emilia_romagna_2023-05-23_path95_desc")
    output_dir = Path("outputs_frame1_20230522_path95_desc")
    work_dir = output_dir / "work"
    dem_name = "SRTM 1Sec HGT"
    pixel_spacing_m = 13.92
    proc_bbox = bbox_centered_pixel_window(
        base=bbox,
        pixel_spacing_m=pixel_spacing_m,
        width_px=target_width_px,
        height_px=target_height_px,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    gpt_path = detect_gpt_path()
    print(f"SNAP GPT: {gpt_path}")

    slc_archives = find_slc_archives(input_dir)
    scenes = identify_scenes(slc_archives)
    if len(scenes) < 2:
        raise RuntimeError("Impossibile identificare almeno due scene SLC valide")

    # Use the most recent two scenes.
    chosen = scenes[-2:]
    chosen.sort(key=lambda p: scene_start_token(p))

    per_date_outputs = []
    per_scene_subswath: dict[str, str] = {}
    subswath_candidates = ["IW1", "IW2", "IW3"]
    for src_zip in chosen:
        date_label = scene_date_label(src_zip)

        grd_raw = work_dir / f"grd_linear_{date_label}.tif"
        grd_db = work_dir / f"grd_db_{date_label}.tif"
        grd_clip = output_dir / f"grd_{date_label}.tif"

        c2_raw = work_dir / f"c2_{date_label}.tif"
        c2_clip = work_dir / f"c2_clipped_{date_label}.tif"
        polcoh_raw = work_dir / f"polcoh_{date_label}.tif"
        polcoh_clip = output_dir / f"polcoh_{date_label}.tif"

        grd_xml = work_dir / f"graph_grd_{date_label}.xml"
        pol_xml = work_dir / f"graph_pol_{date_label}.xml"

        selected_subswath = None
        for sw in subswath_candidates:
            print(f"Provo {sw} per scena {src_zip.name}...")
            try:
                write_graph(
                    build_grd_graph(src_zip, grd_raw, proc_bbox, dem_name, pixel_spacing_m, sw),
                    grd_xml,
                )
                run_gpt(gpt_path, grd_xml)
                linear_to_db(grd_raw, grd_db)
                clip_geotiff_to_bbox(grd_db, grd_clip, proc_bbox, nodata=np.nan)
                if raster_has_valid_values(grd_clip):
                    selected_subswath = sw
                    print(f"Subswath selezionato per {src_zip.name}: {sw}")
                    break
                print(f"{sw} non contiene abbastanza valori validi nel bbox.")
            except Exception as exc:
                print(f"{sw} fallito per {src_zip.name}: {exc}")

        if selected_subswath is None:
            raise RuntimeError(
                f"Nessun subswath valido (IW1/IW2/IW3) trovato per {src_zip.name} nel bbox."
            )

        per_scene_subswath[str(src_zip)] = selected_subswath

        write_graph(
            build_polarimetric_graph(
                src_zip,
                c2_raw,
                proc_bbox,
                selected_subswath,
            ),
            pol_xml,
        )
        run_gpt(gpt_path, pol_xml)
        clip_c2_bands(c2_raw, c2_clip)
        compute_polarimetric_coherence(c2_clip, polcoh_raw)
        clip_geotiff_to_bbox(polcoh_raw, polcoh_clip, proc_bbox, nodata=np.nan)

        per_date_outputs.append(
            {
                "date": date_label,
                "scene": str(src_zip),
                "subswath": selected_subswath,
                "grd": str(grd_clip),
                "polcoh": str(polcoh_clip),
            }
        )

    master = chosen[0]
    slave = chosen[1]
    d1 = scene_date_label(chosen[0])
    d2 = scene_date_label(chosen[1])

    int_raw = work_dir / f"intcoh_{d1}_{d2}.tif"
    int_clip = output_dir / f"intcoh_{d1}_{d2}.tif"
    int_xml = work_dir / f"graph_intcoh_{d1}_{d2}.xml"

    preferred = []
    master_sw = per_scene_subswath.get(str(master))
    slave_sw = per_scene_subswath.get(str(slave))
    if master_sw and master_sw == slave_sw:
        preferred.append(master_sw)
    for sw in subswath_candidates:
        if sw not in preferred:
            preferred.append(sw)

    int_selected_subswath = None
    for sw in preferred:
        print(f"Provo {sw} per coerenza interferometrica...")
        try:
            write_graph(
                build_interferometric_coherence_graph(
                    master_zip=master,
                    slave_zip=slave,
                    out_tif=int_raw,
                    bbox=proc_bbox,
                    dem_name=dem_name,
                    pixel_spacing_m=pixel_spacing_m,
                    subswath=sw,
                ),
                int_xml,
            )
            run_gpt(gpt_path, int_xml)
            clip_geotiff_to_bbox(int_raw, int_clip, proc_bbox, nodata=np.nan)
            if raster_has_valid_values(int_clip):
                int_selected_subswath = sw
                print(f"Subswath selezionato per interferometria: {sw}")
                break
            print(f"{sw} non contiene abbastanza valori validi per interferometria nel bbox.")
        except Exception as exc:
            print(f"{sw} fallito per interferometria: {exc}")

    if int_selected_subswath is None:
        raise RuntimeError("Nessun subswath valido trovato per la coerenza interferometrica nel bbox.")

    metadata = {
        "bbox_input": {
            "min_lon": bbox.min_lon,
            "min_lat": bbox.min_lat,
            "max_lon": bbox.max_lon,
            "max_lat": bbox.max_lat,
        },
        "bbox_processing": {
            "min_lon": proc_bbox.min_lon,
            "min_lat": proc_bbox.min_lat,
            "max_lon": proc_bbox.max_lon,
            "max_lat": proc_bbox.max_lat,
        },
        "target_pixels": {
            "width": target_width_px,
            "height": target_height_px,
        },
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "gpt_path": str(gpt_path),
        "dem_name": dem_name,
        "pixel_spacing_m": pixel_spacing_m,
        "per_date_outputs": per_date_outputs,
        "interferometric_coherence": str(int_clip),
        "interferometric_subswath": int_selected_subswath,
    }
    save_run_metadata(output_dir / "run_metadata.json", metadata)

    # Analyze pixel coverage within bbox for all output rasters
    raster_files_to_analyze = {}
    for item in per_date_outputs:
        raster_files_to_analyze[f"GRD {item['date']}"] = Path(item['grd'])
        raster_files_to_analyze[f"Pol.Coh {item['date']}"] = Path(item['polcoh'])
    raster_files_to_analyze[f"Int.Coh {d1}-{d2}"] = int_clip
    
    analyze_coverage_in_bbox(raster_files_to_analyze, proc_bbox)

    print("\nPipeline completata.")
    for item in per_date_outputs:
        print(f"- GRD: {item['grd']}")
        print(f"- Polarimetric coherence: {item['polcoh']}")
    print(f"- Interferometric coherence: {int_clip}")


if __name__ == "__main__":
    main()
