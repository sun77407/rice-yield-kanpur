"""Canonical daily weather table and Kharif per-district-year features (ADR-003, ADR-026)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
INTERIM = Path("data/interim")
OUT = Path("data/processed/weather_district_year.parquet")
DRY_MM = 1.0  # a day below this is "no rain" for dry spells (CHIRPS has light-drizzle noise)
SPELL_DAYS = 7
GDD_BASE = 10.0


def canonical_daily() -> pd.DataFrame:
    """Daily per-district table: CHIRPS rain, NASA POWER temperature/RH (ERA5 appended later)."""
    chirps = pd.read_parquet(RAW / "chirps/daily.parquet").rename(columns={"precip_mm": "rain_mm"})
    power = pd.read_parquet(RAW / "nasa_power/daily.parquet").rename(
        columns={
            "T2M_MAX": "tmax_c",
            "T2M_MIN": "tmin_c",
            "T2M": "tmean_c",
            "RH2M": "rh_pct",
            "GWETROOT": "gwetroot",
        }
    )[["date", "tmax_c", "tmin_c", "tmean_c", "rh_pct", "gwetroot"]]
    imd = pd.read_parquet(RAW / "imd/daily.parquet").rename(
        columns={"precip_mm": "imd_rain_mm", "tmax_c": "imd_tmax_c", "tmin_c": "imd_tmin_c"}
    )
    out = chirps.merge(power, on="date", how="left").merge(imd, on="date", how="left")
    era5 = RAW / "era5_land/daily.parquet"
    if era5.exists():  # appended after the CDS licence is accepted
        e = pd.read_parquet(era5).add_prefix("era5_").rename(columns={"era5_date": "date"})
        out = out.merge(e, on="date", how="left")
    return out.sort_values(["district", "date"]).reset_index(drop=True)


def _spells(dry: np.ndarray) -> int:
    n = run = 0
    for d in dry:
        run = run + 1 if d else 0
        if run == SPELL_DAYS:
            n += 1
    return n


def district_year(daily: pd.DataFrame) -> pd.DataFrame:
    """Kharif (Jun 1 - Oct 31) aggregates per district and year."""
    rows = []
    for (dist, year), g in daily.groupby(["district", daily.date.dt.year]):
        k = g[(g.date.dt.month >= 6) & (g.date.dt.month <= 10)].sort_values("date")
        jun = k[k.date.dt.month == 6]
        wet = k[k.rain_mm >= 5.0]
        onset = wet.date.iloc[0].dayofyear if len(wet) else np.nan
        flower = k[(k.date >= f"{year}-08-15") & (k.date <= f"{year}-09-15")]
        monthly = k.groupby(k.date.dt.month).rain_mm.sum()
        rows.append(
            {
                "district": dist,
                "year": year,
                "kharif_rainfall_mm": k.rain_mm.sum(),
                "monsoon_onset_day": onset,
                "onset_in_june": bool(len(jun[jun.rain_mm >= 5.0])),
                "dry_spell_count": _spells((k.rain_mm < DRY_MM).to_numpy()),
                "kharif_tmax_mean": k.tmax_c.mean(),
                "kharif_tmin_mean": k.tmin_c.mean(),
                "heat_stress_days": int((flower.tmax_c > 35.0).sum()),
                "kharif_rh_mean": k.rh_pct.mean(),
                "rain_cv": monthly.std(ddof=0) / monthly.mean(),
                "gdd_kharif": float(np.clip((k.tmax_c + k.tmin_c) / 2 - GDD_BASE, 0, None).sum()),
                "kharif_gwetroot_mean": k.gwetroot.mean(),
                "imd_kharif_rainfall_mm": k.imd_rain_mm.sum(),
                "imd_kharif_tmax_mean": k.imd_tmax_c.mean(),
                "imd_kharif_tmin_mean": k.imd_tmin_c.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write the canonical daily table and the district-year features."""
    daily = canonical_daily()
    INTERIM.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(INTERIM / "weather_daily_canonical.parquet", index=False)
    feats = district_year(daily)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, index=False)
    pd.set_option("display.width", 250)
    print(feats.round(2).to_string())


if __name__ == "__main__":
    main()
