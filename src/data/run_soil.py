"""SoilGrids 2.0 (0-5 cm) on the MODIS grid via an EE asset, then per-pixel table (ADR-024)."""

from __future__ import annotations

import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyproj import Transformer

from src.data import satellite as sat
from src.data.manifest import DataManifest

RAW = Path("data/raw/soilgrids")
OUT = Path("data/processed/soil_per_pixel.parquet")
# name -> (asset, band, divisor to reach the output unit, output column)
SPEC = {
    "ph": ("phh2o", 10.0),  # pH x10 -> pH
    "soc": ("soc", 10.0),  # dg/kg -> g/kg
    "clay_pct": ("clay", 10.0),  # g/kg -> %
    "sand_pct": ("sand", 10.0),  # g/kg -> %
    "cec": ("cec", 10.0),  # mmol(c)/kg -> cmol(c)/kg
    # Catalog says cg/kg, but raw median is ~1465 (= 14.7 g/kg if taken at face value, with SOC
    # at 17 g/kg: C:N ~ 1.2, impossible). Read as mg/kg it is 1.47 g/kg (C:N ~ 12), so raw/1e4 is
    # percent N (g per 100 g), the 0.1-0.3 range expected for topsoil. See ADR-028 note.
    "nitrogen": ("nitrogen", 10000.0),
}
YEAR = 2020  # SoilGrids 2.0 release year, used as the asset-path vintage


def main() -> None:
    """Export, download and tabulate soil properties for every pixel in the two districts."""
    load_dotenv()
    project = os.environ["GEE_PROJECT"]
    sat.init_ee(project)
    districts = sat.load_districts()
    region = sat.ee_region(districts)
    crs, transform = sat.modis_grid()
    asset = sat.asset_id(project, "soilgrids", YEAR, "topsoil_0_5cm")
    try:
        ee.data.getAsset(asset)
    except ee.EEException:
        img = ee.Image.cat(
            [
                ee.Image(f"projects/soilgrids-isric/{a}_mean").select(f"{a}_0-5cm_mean").rename(n)
                for n, (a, _) in SPEC.items()
            ]
        ).toInt32()
        job = sat.start_export(
            img.unmask(sat.NODATA),
            "soilgrids_topsoil",
            asset,
            region,
            crs,
            transform,
            0.01,
            "a-priori guess; 6 bands, 1 time, same grid as MODIS stack",
            {"dataset": "projects/soilgrids-isric/*_mean", "depth": "0-5cm"},
        )
        print("EECU-s", sat.wait_and_log(job))
    zp = RAW / "soilgrids_0_5cm.zip"
    if not zp.exists():
        sat.download_asset(asset, region, crs, transform, zp)
    bands, tf = sat.read_band_zip(zp)
    h, w = next(iter(bands.values())).shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs, ys = (np.asarray(a) for a in tf * (cols + 0.5, rows + 0.5))
    lon, lat = Transformer.from_crs(sat.SINU_PROJ4, "EPSG:4326", always_xy=True).transform(xs, ys)
    dist = sat._district_raster(districts, tf, (h, w))
    m = dist > 0
    col_g = np.floor((xs - sat._X0) / sat._PIX).astype(np.int64)
    row_g = np.floor((sat._Y0 - ys) / sat._PIX).astype(np.int64)
    df = pd.DataFrame({"pixel_id": [f"{r}_{c}" for r, c in zip(row_g[m], col_g[m], strict=True)]})
    for n, (_, div) in SPEC.items():
        v = bands[n][m].astype(float)
        v[(v <= sat.NODATA) | (v == np.iinfo(np.int32).min)] = np.nan
        df[n] = v / div
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    DataManifest.build(
        "SoilGrids 2.0 via Google Earth Engine",
        "https://developers.google.com/earth-engine/datasets/catalog/projects_soilgrids-isric_phh2o_mean",
        [zp],
        len(df),
        {
            "depth": "0-5cm",
            "stat": "mean",
            "units": "ph; soc g/kg; clay,sand %; cec cmol/kg; N % (g/100g)",
        },
        {zp.name: asset},
    ).write(RAW / "manifest.json")
    print(df.describe().T.round(2), "\nnull share:\n", df.isna().mean().round(4))


if __name__ == "__main__":
    main()
