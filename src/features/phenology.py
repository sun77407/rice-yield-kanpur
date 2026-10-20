"""Pixel-year phenology from MODIS NDVI (ADR-025: QA<=1, >=4 valid obs, SG order 2 window 5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

INTERIM = Path("data/interim")
OUT = Path("data/processed/pixel_year_phenology.parquet")
QA_MAX = 1
MIN_VALID = 4
SG_WINDOW, SG_ORDER = 5, 2
CROP_MIN = 0.5  # ADR-027: pixel kept if >= 50% of its area is WorldCover cropland
THRESH = 0.4
COMPOSITE_CENTER = 8  # 16-day composite starting at DOY d is centred on d + 8


def _fill_row(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(y)
    return np.asarray(
        np.interp(x, x[ok], y[ok])
    )  # interior gaps linear, edge gaps nearest valid value


def _first_cross(day: np.ndarray, curve: np.ndarray, start: int, up: bool) -> float:
    """First DOY at or after index ``start`` where ``curve`` crosses THRESH (linear sub-step)."""
    for i in range(max(start, 1), len(day)):
        a, b = curve[i - 1], curve[i]
        if (up and a < THRESH <= b) or (not up and a >= THRESH > b):
            return float(day[i - 1] + (THRESH - a) / (b - a) * (day[i] - day[i - 1]))
    return float("nan")


def year_features(
    df: pd.DataFrame, year: int, window: int = SG_WINDOW, order: int = SG_ORDER
) -> pd.DataFrame:
    """Phenology features for one year of pixel-time rows (cropland pixels only)."""
    df = df[df.cropland_frac >= CROP_MIN].sort_values(["pixel_id", "doy"])
    doys = np.sort(df.doy.unique())
    n_t = len(doys)
    n_pix = len(df) // n_t
    ndvi = df.ndvi.to_numpy().reshape(n_pix, n_t).copy()
    qa = df.summary_qa.to_numpy().reshape(n_pix, n_t)
    meta = df.iloc[::n_t][["pixel_id", "district", "lat", "lon", "cropland_frac"]].reset_index(
        drop=True
    )
    ndvi[(qa > QA_MAX) | np.isnan(ndvi)] = np.nan
    n_valid = (~np.isnan(ndvi)).sum(axis=1)
    keep = n_valid >= MIN_VALID
    x = (doys + COMPOSITE_CENTER).astype(float)
    day = np.arange(x[0], x[-1] + 1)
    leap = pd.Timestamp(year, 12, 31).dayofyear == 366
    gf_lo, gf_hi = (228, 259) if leap else (227, 258)  # Aug 15 .. Sep 15

    rows = []
    for k in np.flatnonzero(keep):
        filled = _fill_row(ndvi[k], x)
        sm = savgol_filter(filled, window, order)
        curve = np.interp(day, x, sm)
        ipk = int(np.argmax(curve))
        green = _first_cross(day, curve, 1, up=True)
        sen = _first_cross(day, curve, ipk, up=False)
        gf = curve[(day >= gf_lo) & (day <= gf_hi)]
        rows.append(
            (
                green,
                float(day[ipk]),
                float(curve[ipk]),
                sen,
                sen - green,
                float(np.trapezoid(curve, day)),
                float(gf.mean()),
                float(np.nanmax(ndvi[k])),
                int(n_valid[k]),
                float(np.abs(np.diff(sm, 2)).mean()),
            )
        )
    cols = [
        "greenup_day",
        "peak_ndvi_day",
        "peak_ndvi",
        "senescence_day",
        "length_of_season",
        "integrated_ndvi",
        "grain_filling_ndvi",
        "raw_peak_ndvi",
        "n_valid_obs",
        "smooth_roughness",
    ]
    out = pd.concat([meta[keep].reset_index(drop=True), pd.DataFrame(rows, columns=cols)], axis=1)
    out.insert(2, "year", year)
    out["senescence_censored"] = out.senescence_day.isna()
    out["greenup_censored"] = out.greenup_day.isna()
    out.attrs["dropped"] = int((~keep).sum())
    out.attrs["n_pixels"] = n_pix
    return out


def main() -> None:
    """Build the phenology parquet; ``--suffix ext330`` reads the extended-window pulls."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--window", type=int, default=SG_WINDOW)
    ap.add_argument("--order", type=int, default=SG_ORDER)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    sfx = f"_{a.suffix}" if a.suffix else ""
    parts, stats = [], []
    for y in range(2010, 2025):
        raw = pd.read_parquet(INTERIM / f"modis_ndvi_pixel_{y}{sfx}.parquet")
        res = year_features(raw, y, a.window, a.order)
        parts.append(res)
        stats.append(
            {
                "year": y,
                "pixels_cropland": res.attrs["n_pixels"],
                "dropped_lt4_valid": res.attrs["dropped"],
                "kept": len(res),
                "med_peak_day": res.peak_ndvi_day.median(),
                "med_peak_ndvi": res.peak_ndvi.median(),
                "med_raw_peak_ndvi": res.raw_peak_ndvi.median(),
                "share_peak_ndvi_065_085": res.peak_ndvi.between(0.65, 0.85).mean(),
                "greenup_nan": res.greenup_censored.mean(),
                "senescence_nan": res.senescence_censored.mean(),
                "med_roughness": res.smooth_roughness.median(),
            }
        )
        print(stats[-1], flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    pd.DataFrame(stats).to_csv(out.with_suffix(".checks.csv"), index=False)


if __name__ == "__main__":
    main()
