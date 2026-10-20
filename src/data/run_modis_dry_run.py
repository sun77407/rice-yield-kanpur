"""MODIS dry run for a single year: export to asset, download, convert, report (ADR-022/024)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import ee
from dotenv import load_dotenv

from src.data import satellite as sat

RAW = Path("data/raw/modis")
INTERIM = Path("data/interim")
ESTIMATE_BASIS = "a-priori upper-bound guess (GEE offers no pre-run estimate)"


def _exists(asset: str) -> bool:
    try:
        ee.data.getAsset(asset)
    except ee.EEException:
        return False
    return True


def main() -> None:
    """Run the one-year dry run; reuses assets and downloads that already exist."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    year = ap.parse_args().year
    load_dotenv()
    project = os.environ["GEE_PROJECT"]
    sat.init_ee(project)
    districts = sat.load_districts()
    region = sat.ee_region(districts)
    crs, transform = sat.modis_grid()
    notes: dict[str, Any] = {"crs": crs, "transform": transform, "kharif_doy": sat.KHARIF_DOY}

    crop_asset = sat.asset_id(project, "worldcover", 2021, "cropland_frac_modisgrid")
    year_asset = sat.asset_id(project, "modis_mod13q1", year, "kharif_stack")
    crop_zip = RAW / "cropland_frac_worldcover2021.zip"
    year_zip = RAW / f"modis_mod13q1_{year}.zip"

    jobs: list[sat.ExportJob] = []
    if not _exists(crop_asset):
        jobs.append(
            sat.start_export(
                sat.cropland_fraction(crs, transform, region),
                "cropland_frac_worldcover2021",
                crop_asset,
                region,
                crs,
                transform,
                0.25,
                ESTIMATE_BASIS,
                {"dataset": sat.WORLDCOVER, "districts": list(districts)},
            )
        )
    if not _exists(year_asset):
        stack, doys = sat.year_stack(year, region)
        notes["composite_start_doys"] = doys
        jobs.append(
            sat.start_export(
                stack,
                f"modis_mod13q1_{year}",
                year_asset,
                region,
                crs,
                transform,
                0.10,
                ESTIMATE_BASIS,
                {"collection": sat.MODIS_COLLECTION, "year": year, "doys": doys},
            )
        )
    for j in jobs:
        eecu = sat.wait_and_log(j)
        print(f"{j.export_id}: {eecu} EECU-seconds  asset={j.asset_id}")

    if not crop_zip.exists():
        sat.download_asset(crop_asset, region, crs, transform, crop_zip)
    if not year_zip.exists():
        sat.download_asset(year_asset, region, crs, transform, year_zip)

    df = sat.stack_to_frame(year_zip, crop_zip, districts, year)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"modis_ndvi_pixel_{year}.parquet"
    df.to_parquet(out, index=False)
    ids = {crop_zip.name: crop_asset, year_zip.name: year_asset}
    sat.write_modis_manifest(RAW, [crop_zip, year_zip], len(df), notes, ids)
    print(f"rows={len(df)} pixels={df.pixel_id.nunique()} -> {out}")
    print(df.groupby("district").pixel_id.nunique())


if __name__ == "__main__":
    main()
