"""APY calibration (ADR-002): district yield = alpha + beta * district-mean grain-filling NDVI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROCESSED = Path("data/processed")
APY_CSV = Path(os.environ.get("APY_CSV", "data/raw/apy/apy_kanpur_rice_kharif.csv"))
FEATURE = "grain_filling_ndvi"


def load_apy(path: Path = APY_CSV) -> pd.DataFrame:
    """APY rows for Kanpur Nagar/Dehat Kharif rice; columns district, year, yield_t_ha."""
    df = pd.read_csv(path)
    missing = {"district", "year", "yield_t_ha"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[["district", "year", "yield_t_ha"]]


def calibrate(
    pheno: pd.DataFrame, apy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (district-year targets, pixel-year targets, calibration table)."""
    dy = (
        pheno.groupby(["district", "year"])
        .agg(mean_gf_ndvi=(FEATURE, "mean"), n_pixels=("pixel_id", "nunique"))
        .reset_index()
        .merge(
            apy.rename(columns={"yield_t_ha": "apy_yield_t_ha"}),
            on=["district", "year"],
            how="left",
        )
    )
    cal = []
    for dist, g in dy.dropna(subset=["apy_yield_t_ha"]).groupby("district"):
        x, y = g.mean_gf_ndvi.to_numpy(), g.apy_yield_t_ha.to_numpy()
        res = stats.linregress(x, y)
        # leave-one-year-out R2: honest out-of-sample skill with N~13
        pred = np.array(
            [
                np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i])
                for i in range(len(x))
            ]
        )
        loo_r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        cal.append(
            {
                "district": dist,
                "n_years": len(g),
                "alpha": res.intercept,
                "beta": res.slope,
                "r2": res.rvalue**2,
                "p_value": res.pvalue,
                "loo_r2": loo_r2,
                "rmse_t_ha": float(np.sqrt(((y - (res.intercept + res.slope * x)) ** 2).mean())),
            }
        )
    cal_df = pd.DataFrame(cal)
    ab = cal_df[["district", "alpha", "beta"]]
    dy = dy.merge(ab, on="district", how="left")
    dy["calibrated_yield_t_ha"] = dy.alpha + dy.beta * dy.mean_gf_ndvi
    px = pheno[["pixel_id", "district", "year", FEATURE]].merge(ab, on="district", how="left")
    px["yield_est_t_ha"] = px.alpha + px.beta * px[FEATURE]
    return dy, px[["pixel_id", "district", "year", "yield_est_t_ha"]], cal_df


def main() -> None:
    """Write targets_district_year / targets_pixel_year and the calibration table."""
    pheno = pd.read_parquet(PROCESSED / "pixel_year_phenology.parquet")
    dy, px, cal = calibrate(pheno, load_apy())
    dy.to_parquet(PROCESSED / "targets_district_year.parquet", index=False)
    px.to_parquet(PROCESSED / "targets_pixel_year.parquet", index=False)
    cal.to_csv(PROCESSED / "apy_calibration.csv", index=False)
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()
