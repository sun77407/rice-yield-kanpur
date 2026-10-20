"""Phase 3 shared data loading, split, metrics and district aggregation (ADR-028..030)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.targets import load_apy
from src.models.feature_sets import FORBIDDEN
from src.utils.seed import GLOBAL_SEED

PROCESSED = Path("data/processed")
RESULTS = Path("models/results")
PREDS = Path("data/processed/preds")
Y_MIN, Y_MAX = 0.5, 5.0  # ADR-029
TRAIN_YEARS = tuple(range(2010, 2021))
TEST_YEARS = tuple(range(2021, 2025))
DISTRICTS = ("KanpurDehat", "KanpurNagar")


def load_matrix() -> pd.DataFrame:
    """X_pixel_year with a clipped target ``y`` (raw kept as ``y_raw``)."""
    x = pd.read_parquet(PROCESSED / "X_pixel_year.parquet")
    x["y_raw"] = x["yield_est_t_ha"]
    x["y"] = x["y_raw"].clip(Y_MIN, Y_MAX)
    return x


def clip_stats(x: pd.DataFrame) -> pd.DataFrame:
    """Share of raw targets below/above the clip bounds, overall, per district and per split."""
    rows = []
    groups: dict[str, pd.Series] = {
        "all": pd.Series(True, index=x.index),
        "train_2010_2020": x.year.isin(TRAIN_YEARS),
        "test_2021_2024": x.year.isin(TEST_YEARS),
    }
    for d in DISTRICTS:
        groups[d] = x.district == d
    for name, m in groups.items():
        r = x.loc[m, "y_raw"]
        rows.append(
            {
                "subset": name,
                "n": len(r),
                "pct_below_0.5": 100 * (r < Y_MIN).mean(),
                "pct_above_5.0": 100 * (r > Y_MAX).mean(),
                "pct_clipped": 100 * ((r < Y_MIN) | (r > Y_MAX)).mean(),
                "raw_min": r.min(),
                "raw_p01": r.quantile(0.01),
                "raw_median": r.median(),
                "raw_p99": r.quantile(0.99),
                "raw_max": r.max(),
            }
        )
    return pd.DataFrame(rows)


def split(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spatiotemporal split: train 2010-2020, test 2021-2024 (no year shared)."""
    tr, te = x[x.year.isin(TRAIN_YEARS)], x[x.year.isin(TEST_YEARS)]
    assert not set(tr.year) & set(te.year)
    return tr, te


def check_features(cols: list[str]) -> list[str]:
    """Guard against leaky features (ADR-028)."""
    bad = FORBIDDEN & set(cols)
    if bad:
        raise ValueError(f"forbidden features requested: {sorted(bad)}")
    return cols


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """R2, RMSE, MAE."""
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def metrics_by_district(df: pd.DataFrame, pred_col: str = "pred") -> dict[str, dict[str, float]]:
    """Return combined and per-district metrics for a frame with ``y`` and ``pred``."""
    out = {"combined": metrics(df.y.to_numpy(), df[pred_col].to_numpy())}
    for d in DISTRICTS:
        s = df[df.district == d]
        out[d] = metrics(s.y.to_numpy(), s[pred_col].to_numpy())
    return out


def district_agg(df: pd.DataFrame, pred_col: str = "pred") -> dict[str, Any]:
    """Aggregate pixel predictions to district-year means and compare with APY yields.

    Returns R2/RMSE against APY (a) over every APY year in ``df`` and (b) per district. Years
    that were in the training split are in-sample; the caller decides which years ``df`` holds.
    """
    apy = load_apy().rename(columns={"yield_t_ha": "apy"})
    agg = (
        df.groupby(["district", "year"], as_index=False)[pred_col]
        .mean()
        .merge(apy, on=["district", "year"])
    )
    if len(agg) < 3:
        return {"n": len(agg)}
    out: dict[str, Any] = {
        "n": len(agg),
        "pooled": {
            **metrics(agg.apy.to_numpy(), agg[pred_col].to_numpy()),
            "pearson_r": float(np.corrcoef(agg.apy, agg[pred_col])[0, 1]),
        },
    }
    for d in DISTRICTS:
        s = agg[agg.district == d]
        if len(s) >= 3:
            out[d] = {
                **metrics(s.apy.to_numpy(), s[pred_col].to_numpy()),
                "pearson_r": float(np.corrcoef(s.apy, s[pred_col])[0, 1]),
                "n": len(s),
            }
    return out


def save_preds(name: str, df: pd.DataFrame, pred: np.ndarray) -> None:
    """Persist test-split predictions for the report."""
    PREDS.mkdir(parents=True, exist_ok=True)
    out = df[["pixel_id", "district", "year", "y", "y_raw"]].copy()
    out["pred"] = pred
    out.to_parquet(PREDS / f"{name}.parquet", index=False)


def seed() -> int:
    """Global seed."""
    return GLOBAL_SEED
