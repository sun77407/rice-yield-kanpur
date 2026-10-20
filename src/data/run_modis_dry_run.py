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

    # 10 m WorldCover reduceResolution fails above ~20M source px per export (probed: a
    # 0.3x0.5 degree box fails, 0.15x0.15 works), so tile the box into 4x4 ~0.27 degree cells.
    tiles = sat.tile_regions(region, 4, 4)
    n_tiles = len(tiles)
    crop_ids = [f"cropland_frac_t{i}" for i in range(n_tiles)]
    crop_assets = [sat.asset_id(project, "worldcover", 2021, cid) for cid in crop_ids]
    year_asset = sat.asset_id(project, "modis_mod13q1", year, "kharif_stack")
    crop_zips = [RAW / f"{cid}_worldcover2021.zip" for cid in crop_ids]
    year_zip = RAW / f"modis_mod13q1_{year}.zip"

    jobs: list[sat.ExportJob] = []
    running = sat.adopt_running(f"modis_mod13q1_{year}", year_asset)
    if running:
        print(f"adopting in-flight task {running.task_id}")
        jobs.append(running)
    for cid, asset, tile in zip(crop_ids, crop_assets, tiles, strict=True):
        if (live := sat.adopt_running(cid, asset)) is not None:
            jobs.append(live)
        elif not _exists(asset):
            jobs.append(
                sat.start_export(
                    sat.cropland_fraction(tile),
                    cid,
                    asset,
                    tile,
                    crs,
                    transform,
                    0.25 / n_tiles,
                    ESTIMATE_BASIS,
                    {"dataset": sat.WORLDCOVER, "districts": list(districts)},
                )
            )
    if not running and not _exists(year_asset):
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
    failures = []
    for j in jobs:
        try:
            eecu = sat.wait_and_log(j)
            print(f"{j.export_id}: {eecu} EECU-seconds  asset={j.asset_id}")
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("export failed: " + "; ".join(failures))

    for zp, asset, tile in zip(crop_zips, crop_assets, tiles, strict=True):
        if not zp.exists():
            sat.download_asset(asset, tile, crs, transform, zp)
    if not year_zip.exists():
        sat.download_asset(year_asset, region, crs, transform, year_zip)

    df = sat.stack_to_frame(year_zip, crop_zips, districts, year)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"modis_ndvi_pixel_{year}.parquet"
    df.to_parquet(out, index=False)
    ids = {zp.name: a for zp, a in zip(crop_zips, crop_assets, strict=True)}
    ids[year_zip.name] = year_asset
    sat.write_modis_manifest(RAW, [*crop_zips, year_zip], len(df), notes, ids)
    print(f"rows={len(df)} pixels={df.pixel_id.nunique()} -> {out}")
    print(df.groupby("district").pixel_id.nunique())


if __name__ == "__main__":
    main()
