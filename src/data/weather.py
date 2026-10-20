"""Weather acquisition: NASA POWER, CHIRPS (via GEE), ERA5-Land (CDS), IMD (imdlib).

Error doctrine (PROJECT_FLOW.md section 5): 3 attempts with exponential backoff, every
attempt/failure logged as JSON lines to ``logs/data_acquisition.jsonl``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import requests

from src.data.manifest import DataManifest

ACQ_LOG = Path("logs/data_acquisition.jsonl")
RAW = Path("data/raw")
START, END = "2010-01-01", "2024-12-31"
KANPUR_LAT, KANPUR_LON = 26.4499, 80.3319
# bounding box of the two GADM districts plus a margin, [W, S, E, N]
BBOX = (79.4, 25.9, 80.7, 27.2)
POWER_PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PS,GWETROOT,GWETTOP,ALLSKY_SFC_SW_DWN"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
T = TypeVar("T")


def log_event(source: str, event: str, **fields: Any) -> None:
    """Append a structured record to ``logs/data_acquisition.jsonl``."""
    ACQ_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "source": source}
    with ACQ_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**rec, "event": event, **fields}, default=str) + "\n")


def with_retries(source: str, what: str, fn: Callable[[], T], attempts: int = 3) -> T:
    """Run ``fn`` with exponential backoff (2 s, 4 s); log each failure; re-raise the last."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            log_event(
                source, "retry" if i < attempts else "failed", what=what, attempt=i, error=repr(exc)
            )
            if i == attempts:
                raise
            time.sleep(2**i)
    raise AssertionError("unreachable")


def _write_manifest(
    source: str, url: str, files: list[Path], rows: int, notes: dict[str, Any]
) -> None:
    out = RAW / source
    DataManifest.build(source, url, files, rows, notes).write(out / "manifest.json")


def fetch_nasa_power() -> pd.DataFrame:
    """Daily NASA POWER (AG community) at the Kanpur centroid, 2010-2024."""
    params = {
        "parameters": POWER_PARAMS,
        "community": "AG",
        "longitude": KANPUR_LON,
        "latitude": KANPUR_LAT,
        "start": START.replace("-", ""),
        "end": END.replace("-", ""),
        "format": "JSON",
    }

    def call() -> dict[str, Any]:
        r = requests.get(POWER_URL, params=params, timeout=180)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        return data

    js = with_retries("nasa_power", "daily point 2010-2024", call)
    par = js["properties"]["parameter"]
    df = pd.DataFrame(par)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename_axis("date").reset_index()
    df = df.replace(-999.0, np.nan)
    out = RAW / "nasa_power" / "daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    nfill = {c: int(df[c].isna().sum()) for c in df.columns if c != "date"}
    _write_manifest(
        "nasa_power",
        r_url(params),
        [out],
        len(df),
        {"point": [KANPUR_LAT, KANPUR_LON], "fill_value_-999_to_nan": nfill, "failures": []},
    )
    return df


def r_url(params: dict[str, Any]) -> str:
    """Request URL for provenance."""
    return requests.Request("GET", POWER_URL, params=params).prepare().url or POWER_URL


def fetch_chirps(project: str) -> pd.DataFrame:
    """CHIRPS v2 daily rainfall, mean over each GADM district polygon, via Earth Engine."""
    import ee

    from src.data import satellite as sat
    from src.data.gee_quota import log_actual, log_estimate

    sat.init_ee(project)
    districts = sat.load_districts()
    fc = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry(g, None, False), {"district": name})
            for name, g in districts.items()
        ]
    )
    col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    frames: list[pd.DataFrame] = []
    for year in range(int(START[:4]), int(END[:4]) + 1):
        eid = f"chirps_daily_{year}"
        log_estimate(
            eid, 0.002, "interactive reduceRegions, 365 daily images x 2 polygons", {"year": year}
        )

        def per_image(img: Any) -> Any:
            red = img.reduceRegions(fc, ee.Reducer.mean(), 5566)
            date = img.date().format("YYYY-MM-dd")
            return red.map(lambda f: f.set("date", date))

        def call(y: int = year) -> list[dict[str, Any]]:
            sub = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
            res = sub.map(per_image).flatten().getInfo()
            return [f["properties"] for f in res["features"]]

        props = with_retries("chirps", f"year {year}", call)
        log_actual(eid, "interactive", None, "COMPLETED")  # interactive calls are not metered
        frames.append(pd.DataFrame(props))
        print("chirps", year, len(props))
    df = pd.concat(frames, ignore_index=True).rename(columns={"mean": "precip_mm"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "district", "precip_mm"]].sort_values(["date", "district"])
    out = RAW / "chirps" / "daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _write_manifest(
        "chirps",
        "https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY",
        [out],
        len(df),
        {"reducer": "mean over GADM district polygon, scale 5566 m", "failures": []},
    )
    return df


def fetch_era5_month(year: int, month: int, out_dir: Path) -> Path:
    """One month of hourly ERA5-Land (4 variables, Kanpur box) as NetCDF via the CDS API."""
    import cdsapi

    target = out_dir / f"era5land_{year}{month:02d}.nc"
    if target.exists():
        return target
    w, s, e, n = BBOX
    req = {
        "variable": [
            "2m_temperature",
            "total_precipitation",
            "2m_dewpoint_temperature",
            "surface_pressure",
        ],
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [n, w, s, e],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")

    def call() -> Path:
        c = cdsapi.Client(url=url, key=key, quiet=True)
        tmp = target.with_suffix(".part")
        c.retrieve("reanalysis-era5-land", req, str(tmp))
        tmp.replace(target)
        return target

    return with_retries("era5_land", f"{year}-{month:02d}", call)


def era5_daily(files: list[Path]) -> pd.DataFrame:
    """Aggregate hourly NetCDFs to UTC-day values averaged over the box.

    ERA5-Land ``tp`` accumulates from 00 UTC; the daily total of day d is the value at 00 UTC
    of d+1, so total precipitation is taken from hour 0 and shifted back one day.
    """
    import xarray as xr

    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        tcoord = "valid_time" if "valid_time" in ds.coords else "time"
        ds = ds.mean(dim=[d for d in ("latitude", "longitude") if d in ds.dims])
        df = ds.to_dataframe().reset_index().rename(columns={tcoord: "time"})
        parts.append(df[["time", "t2m", "d2m", "sp", "tp"]])
        ds.close()
    h = pd.concat(parts).drop_duplicates("time").sort_values("time")
    h["date"] = h["time"].dt.normalize()
    daily = h.groupby("date").agg(
        t2m_mean_c=("t2m", lambda s: s.mean() - 273.15),
        t2m_max_c=("t2m", lambda s: s.max() - 273.15),
        t2m_min_c=("t2m", lambda s: s.min() - 273.15),
        d2m_mean_c=("d2m", lambda s: s.mean() - 273.15),
        sp_mean_pa=("sp", "mean"),
    )
    tp0 = h[h["time"].dt.hour == 0].copy()
    tp0["date"] = tp0["date"] - pd.Timedelta(days=1)
    daily["tp_mm"] = tp0.set_index("date")["tp"] * 1000.0
    return daily.reset_index()
