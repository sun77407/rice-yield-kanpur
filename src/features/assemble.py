"""Join satellite + weather + soil + targets; write X matrices and diagnostics (Agent 2D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.feature_sets import FULL

P = Path("data/processed")
FIG = Path("figures/phase2_diagnostics.png")
TARGET = "yield_est_t_ha"
PHENO = [
    "greenup_day",
    "peak_ndvi_day",
    "peak_ndvi",
    "senescence_day",
    "length_of_season",
    "integrated_ndvi",
    "grain_filling_ndvi",
]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_pixel_year, X_district_year)."""
    ph = pd.read_parquet(P / "pixel_year_phenology.parquet")
    wx = pd.read_parquet(P / "weather_district_year.parquet")
    soil = pd.read_parquet(P / "soil_per_pixel.parquet")
    tpx = pd.read_parquet(P / "targets_pixel_year.parquet")
    tdy = pd.read_parquet(P / "targets_district_year.parquet")
    x = (
        ph.merge(wx, on=["district", "year"], how="left")
        .merge(soil, on="pixel_id", how="left")
        .merge(tpx, on=["pixel_id", "district", "year"], how="left")
    )
    # ADR-028: calibration-derived / target-correlated NDVI features never enter the matrix
    x = x.drop(columns=["grain_filling_ndvi", "integrated_ndvi", "peak_ndvi", "smooth_roughness"])
    x["district_kanpur_dehat"] = (x.district == "KanpurDehat").astype(int)
    x["district_kanpur_nagar"] = (x.district == "KanpurNagar").astype(int)
    sat_dy = ph.groupby(["district", "year"])[PHENO].mean().add_prefix("mean_").reset_index()
    xd = (
        tdy.merge(wx, on=["district", "year"], how="left")
        .merge(sat_dy, on=["district", "year"], how="left")
        .dropna(subset=["apy_yield_t_ha"])
        .reset_index(drop=True)
    )
    return x, xd


def vif(df: pd.DataFrame) -> pd.Series:
    """VIF from the inverse correlation matrix (median-imputed, constants dropped)."""
    d = df.fillna(df.median())
    d = d.loc[:, d.std() > 0]
    inv = np.linalg.pinv(np.corrcoef(d.to_numpy(), rowvar=False))
    return pd.Series(np.diag(inv), index=d.columns).sort_values(ascending=False)


def diagnostics(x: pd.DataFrame, xd: pd.DataFrame) -> dict[str, Any]:
    """Null counts, duplicate check, correlations, VIF; save the figure."""
    feats = [c for c in FULL if c in x.columns]  # leaky NDVI features excluded (ADR-028)
    nulls = x.isna().mean().mul(100).round(2)
    dup = int(x.duplicated(["pixel_id", "year"]).sum())
    corr = x[feats].corr()
    iu = np.triu_indices_from(corr, 1)
    pairs = pd.DataFrame(
        {"a": corr.columns[iu[0]], "b": corr.columns[iu[1]], "r": corr.to_numpy()[iu]}
    )
    top = pairs.reindex(pairs.r.abs().sort_values(ascending=False).index)
    top = top[top.r.abs() > 0.9].head(5)
    v = vif(x[feats])
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    nn = nulls[nulls > 0].sort_values()
    ax[0, 0].barh(nn.index, nn.values)
    ax[0, 0].set_title("Null % per column (columns with nulls)")
    im = ax[0, 1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[0, 1].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=5)
    ax[0, 1].set_yticks(range(len(corr)), corr.columns, fontsize=5)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("Correlation (pixel-year)")
    vv = v.head(20)[::-1]
    ax[1, 0].barh(vv.index, vv.values)
    ax[1, 0].axvline(10, color="k", ls="--")
    ax[1, 0].set_title("Top-20 VIF (dashed = 10)")
    for d, g in xd.groupby("district"):
        ax[1, 1].scatter(g.mean_gf_ndvi, g.apy_yield_t_ha, label=d)
    ax[1, 1].set_xlabel("district mean grain-filling NDVI")
    ax[1, 1].set_ylabel("APY yield (t/ha)")
    ax[1, 1].legend()
    ax[1, 1].set_title("APY calibration")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=130)
    return {"nulls": nulls, "dups": dup, "top_pairs": top, "vif": v, "corr": corr}


def main() -> None:
    """Write X_pixel_year / X_district_year and print the diagnostics."""
    x, xd = build()
    x.to_parquet(P / "X_pixel_year.parquet", index=False)
    xd.to_parquet(P / "X_district_year.parquet", index=False)
    d = diagnostics(x, xd)
    d["corr"].to_csv(P / "correlation_matrix.csv")
    d["vif"].to_csv(P / "vif.csv")
    print("X_pixel_year", x.shape, "X_district_year", xd.shape, "dup pixel-year", d["dups"])
    print(d["nulls"].to_string())
    print(d["top_pairs"].to_string())
    print(d["vif"].head(15).round(1).to_string())


if __name__ == "__main__":
    main()
