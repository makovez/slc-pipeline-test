from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ReferenceOffset:
    offset_days: int
    search_window_days: int = 20


@dataclass
class ASFConfig:
    username: str | None = None
    password: str | None = None
    username_env: str = "ASF_USERNAME"
    password_env: str = "ASF_PASSWORD"

    def resolve_credentials(self) -> tuple[str, str]:
        username = self.username or os.getenv(self.username_env)
        password = self.password or os.getenv(self.password_env)
        if not username or not password:
            raise ValueError(
                "ASF credentials mancanti: configura asf.username/password nel config "
                "oppure esporta ASF_USERNAME e ASF_PASSWORD."
            )
        return username, password


@dataclass
class DownloadConfig:
    output_dir: Path = Path("downloads")
    max_results_event: int = 200
    max_results_reference: int = 300
    parallel_processes: int = 3


@dataclass
class SnapConfig:
    gpt_path: str | None = None
    dem_name: str = "SRTM 1Sec HGT"
    pixel_spacing_m: float = 13.92
    subswaths: list[str] = field(default_factory=lambda: ["IW1", "IW2", "IW3"])


@dataclass
class PipelineConfig:
    bbox: BBox
    event_date_range: DateRange
    asf: ASFConfig = field(default_factory=ASFConfig)
    snap: SnapConfig = field(default_factory=SnapConfig)
    references: list[ReferenceOffset] = field(
        default_factory=lambda: [ReferenceOffset(offset_days=30), ReferenceOffset(offset_days=40)]
    )
    download: DownloadConfig = field(default_factory=DownloadConfig)

    @classmethod
    def from_file(cls, path: Path) -> "PipelineConfig":
        path = path.expanduser().resolve()
        raw = path.read_text(encoding="utf-8")

        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("PyYAML non installato, impossibile leggere YAML") from exc
            payload = yaml.safe_load(raw)
        else:
            payload = json.loads(raw)

        if not isinstance(payload, dict):
            raise ValueError("Config non valida: il root deve essere un oggetto/dict")

        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        bbox_raw = payload.get("bbox")
        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            bbox = BBox(*[float(x) for x in bbox_raw])
        elif isinstance(bbox_raw, dict):
            bbox = BBox(
                min_lon=float(bbox_raw["min_lon"]),
                min_lat=float(bbox_raw["min_lat"]),
                max_lon=float(bbox_raw["max_lon"]),
                max_lat=float(bbox_raw["max_lat"]),
            )
        else:
            raise ValueError("bbox non valido: usa lista [min_lon,min_lat,max_lon,max_lat] o dict")

        event_raw = payload.get("event_date_range") or {}
        if not isinstance(event_raw, dict):
            raise ValueError("event_date_range non valido")

        event_range = DateRange(
            start=_parse_iso_utc(event_raw["start"]),
            end=_parse_iso_utc(event_raw["end"]),
        )

        asf_raw = payload.get("asf", {})
        if not isinstance(asf_raw, dict):
            raise ValueError("asf non valido")
        asf_cfg = ASFConfig(
            username=asf_raw.get("username"),
            password=asf_raw.get("password"),
            username_env=str(asf_raw.get("username_env", "ASF_USERNAME")),
            password_env=str(asf_raw.get("password_env", "ASF_PASSWORD")),
        )

        refs_raw = payload.get("references", [])
        if not refs_raw:
            refs = [ReferenceOffset(offset_days=30), ReferenceOffset(offset_days=40)]
        else:
            refs = []
            for item in refs_raw:
                if not isinstance(item, dict):
                    raise ValueError("references deve contenere oggetti")
                refs.append(
                    ReferenceOffset(
                        offset_days=int(item["offset_days"]),
                        search_window_days=int(item.get("search_window_days", 20)),
                    )
                )

        dl_raw = payload.get("download", {})
        if not isinstance(dl_raw, dict):
            raise ValueError("download non valido")
        dl_cfg = DownloadConfig(
            output_dir=Path(str(dl_raw.get("output_dir", "downloads"))).expanduser(),
            max_results_event=int(dl_raw.get("max_results_event", 200)),
            max_results_reference=int(dl_raw.get("max_results_reference", 300)),
            parallel_processes=max(1, int(dl_raw.get("parallel_processes", 3))),
        )

        snap_raw = payload.get("snap", {})
        if not isinstance(snap_raw, dict):
            raise ValueError("snap non valido")
        subswaths = snap_raw.get("subswaths", ["IW1", "IW2", "IW3"])
        if not isinstance(subswaths, list):
            raise ValueError("snap.subswaths non valido")
        snap_cfg = SnapConfig(
            gpt_path=snap_raw.get("gpt_path"),
            dem_name=str(snap_raw.get("dem_name", "SRTM 1Sec HGT")),
            pixel_spacing_m=float(snap_raw.get("pixel_spacing_m", 13.92)),
            subswaths=[str(s) for s in subswaths],
        )

        return cls(
            bbox=bbox,
            event_date_range=event_range,
            asf=asf_cfg,
            snap=snap_cfg,
            references=refs,
            download=dl_cfg,
        )


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
