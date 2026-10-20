"""Run weather acquisition (Agent 1B): NASA POWER, CHIRPS, ERA5-Land, then IMD."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import pandas as pd
from dotenv import load_dotenv

from src.data import weather as w
from src.utils.net import force_ipv4


def era5() -> None:
    """Download ERA5-Land per year (parallel), aggregate to daily, write parquet + manifest."""
    out_dir = w.RAW / "era5_land"
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(2010, 2025))
    with ThreadPoolExecutor(max_workers=2) as ex:
        files = list(ex.map(lambda y: w.fetch_era5_year(y, out_dir), years))
    tail = out_dir / "era5land_20250101.nc"  # 00 UTC on 2025-01-01 closes the 2024-12-31 total
    if not tail.exists():
        lo, so, ea, no = w.BBOX[0], w.BBOX[1], w.BBOX[2], w.BBOX[3]

        def call() -> Path:
            c = cdsapi.Client(
                url=os.environ.get("CDSAPI_URL"), key=os.environ.get("CDSAPI_KEY"), quiet=True
            )
            c.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": ["total_precipitation", "2m_temperature"],
                    "year": "2025",
                    "month": "01",
                    "day": "01",
                    "time": ["00:00"],
                    "area": [no, lo, so, ea],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(tail),
            )
            return tail

        w.with_retries("era5_land", "tail 2025-01-01", call)
    daily = w.era5_daily([*files, tail])
    daily = daily[(daily.date >= w.START) & (daily.date <= w.END)]
    out = out_dir / "daily.parquet"
    daily.to_parquet(out, index=False)
    w._write_manifest(
        "era5_land",
        "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land",
        [out, *files, tail],
        len(daily),
        {
            "bbox_WSEN": w.BBOX,
            "aggregation": "UTC day; box-mean; tp = value at 00 UTC of d+1",
            "failures": [],
        },
    )
    print("era5_land rows", len(daily))


def main() -> None:
    """Dispatch the selected sources."""
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", choices=["power", "chirps", "era5", "imd"])
    args = ap.parse_args()
    load_dotenv()
    force_ipv4()
    for s in args.sources:
        try:
            if s == "power":
                df = w.fetch_nasa_power()
                print("power rows", len(df))
            elif s == "chirps":
                df = w.fetch_chirps(os.environ["GEE_PROJECT"])
                print("chirps rows", len(df))
            elif s == "era5":
                era5()
            else:
                from src.data import imd

                imd.fetch_imd()
        except Exception as exc:
            w.log_event(s, "source_failed", error=repr(exc))
            print(f"{s} FAILED: {exc!r}")


if __name__ == "__main__":
    pd.options.display.width = 140
    main()
