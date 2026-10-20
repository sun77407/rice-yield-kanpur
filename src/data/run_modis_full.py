"""Full-range MODIS export (default 2010-2024): one EE asset per year, reusing cropland tiles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.data import satellite as sat
from src.data.manifest import sha256_file
from src.data.run_modis_dry_run import ESTIMATE_BASIS, INTERIM, RAW, _exists

N_TILES = 16
YEAR_ESTIMATE_H = 0.01  # dry run measured 11.2 EECU-s (0.003 h); keep a 3x margin


def main() -> None:
    """Submit every missing year, wait for all, download, convert, write one manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--doy-end", type=int, default=None, help="extend window (ADR-025 addendum)")
    ap.add_argument("--tag", default="", help="suffix for assets/files, e.g. ext330")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    years = list(range(args.start, args.end + 1))
    load_dotenv()
    project = os.environ["GEE_PROJECT"]
    sat.init_ee(project)
    districts = sat.load_districts()
    region = sat.ee_region(districts)
    crs, transform = sat.modis_grid()

    crop_ids = [f"cropland_frac_t{i}" for i in range(N_TILES)]
    crop_zips = [RAW / f"{cid}_worldcover2021.zip" for cid in crop_ids]
    crop_assets = {
        z.name: sat.asset_id(project, "worldcover", 2021, cid)
        for z, cid in zip(crop_zips, crop_ids, strict=True)
    }
    missing = [z.name for z in crop_zips if not z.exists()]
    if missing:
        raise SystemExit(f"cropland tiles missing locally (not re-exporting): {missing}")

    assets = {y: sat.asset_id(project, "modis_mod13q1", y, f"kharif_stack{sfx}") for y in years}
    doys_by_year: dict[int, list[int]] = {}
    jobs: list[sat.ExportJob] = []
    for y in years:
        eid = f"modis_mod13q1_{y}{sfx}"
        if (live := sat.adopt_running(eid, assets[y])) is not None:
            jobs.append(live)
        elif not _exists(assets[y]):
            stack, doys = sat.year_stack(y, region, args.doy_end)
            doys_by_year[y] = doys
            jobs.append(
                sat.start_export(
                    stack,
                    eid,
                    assets[y],
                    region,
                    crs,
                    transform,
                    YEAR_ESTIMATE_H,
                    ESTIMATE_BASIS + "; 2020 dry run measured 11.2 EECU-s",
                    {"collection": sat.MODIS_COLLECTION, "year": y, "doys": doys},
                )
            )
    print(f"submitted/adopted {len(jobs)} tasks; {len(years) - len(jobs)} already existed")

    failures: list[str] = []
    for j in jobs:
        try:
            print(f"{j.export_id}: {sat.wait_and_log(j)} EECU-s")
        except (RuntimeError, TimeoutError) as exc:
            failures.append(str(exc))
    ok_years = [y for y in years if _exists(assets[y])]

    INTERIM.mkdir(parents=True, exist_ok=True)
    year_info: dict[str, Any] = {}
    zips: list[Path] = []
    for y in ok_years:
        zp = RAW / f"modis_mod13q1_{y}{sfx}.zip"
        if not zp.exists():
            sat.download_asset(assets[y], region, crs, transform, zp)
        zips.append(zp)
        out = INTERIM / f"modis_ndvi_pixel_{y}{sfx}.parquet"
        df = sat.stack_to_frame(zp, [RAW / n for n in crop_assets], districts, y)
        df.to_parquet(out, index=False)
        crop = df[df.cropland_frac >= 0.5]
        year_info[str(y)] = {
            "parquet": str(out),
            "parquet_sha256": sha256_file(out),
            "parquet_bytes": out.stat().st_size,
            "rows": len(df),
            "pixels_by_district": df.groupby("district").pixel_id.nunique().to_dict(),
            "cropland_ge_0.5_pixels_by_district": crop.groupby("district")
            .pixel_id.nunique()
            .to_dict(),
            "share_qa_le_1_kharif": float((df.summary_qa <= 1).mean()),
            "share_qa_eq_0": float((df.summary_qa == 0).mean()),
            "share_qa_eq_3": float((df.summary_qa == 3).mean()),
            "composite_start_doys": sorted(int(d) for d in df.doy.unique()),
        }
        print(y, year_info[str(y)]["rows"], year_info[str(y)]["share_qa_le_1_kharif"])
    ids = {n: a for n, a in crop_assets.items()}
    ids.update({z.name: assets[y] for z, y in zip(zips, ok_years, strict=True)})
    total_rows = sum(v["rows"] for v in year_info.values())
    sat.write_modis_manifest(
        RAW,
        [*(RAW / n for n in crop_assets), *zips],
        total_rows,
        {"years_ok": ok_years, "failures": failures, "per_year": year_info},
        ids,
        name=f"manifest{sfx}.json",
    )
    pd.DataFrame(
        [
            {"year": y, **{k: v for k, v in i.items() if not isinstance(v, dict | list)}}
            for y, i in year_info.items()
        ]
    ).to_csv(INTERIM / f"modis_year_summary{sfx}.csv", index=False)
    if failures:
        raise SystemExit("failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
