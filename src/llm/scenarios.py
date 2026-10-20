"""Scenario inputs for the advisory: historical analog district-years (ADR-032).

The 30 scenarios are the 30 district-years of 2010-2024. Each is labelled bad / average / good
from its own weather (not from yield) and carries the XGBoost district-aggregated yield estimate
(in-sample for 2010-2020) and a per-district linear time-trend reference. They are analogs of
past seasons, not forecasts.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.targets import load_apy
from src.llm.advisor import YieldPrediction
from src.models.feature_sets import FULL

PROCESSED = Path("data/processed")
CACHE = PROCESSED / "district_year_predictions.parquet"


def district_year_predictions() -> pd.DataFrame:
    """Mean XGBoost prediction per district-year over all cropland pixels (cached)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    x = pd.read_parquet(PROCESSED / "X_pixel_year.parquet", columns=["district", "year", *FULL])
    with Path("models/xgboost.pkl").open("rb") as fh:
        model = pickle.load(fh)
    x["pred"] = model.predict(x[FULL])
    out = x.groupby(["district", "year"], as_index=False)[["pred"]].mean()
    out.to_parquet(CACHE, index=False)
    return pd.DataFrame(out)


def classify(w: pd.DataFrame) -> pd.Series:
    """Label each district-year bad / good / average from its own weather.

    bad: rainfall in the lowest third of the district's years or >= 10 heat-stress days;
    good: rainfall in the highest third and no heat stress; otherwise average.
    """
    lo = w.groupby("district").kharif_rainfall_mm.transform(lambda s: s.quantile(1 / 3))
    hi = w.groupby("district").kharif_rainfall_mm.transform(lambda s: s.quantile(2 / 3))
    bad = (w.kharif_rainfall_mm <= lo) | (w.heat_stress_days >= 10)
    good = (w.kharif_rainfall_mm >= hi) & (w.heat_stress_days == 0) & ~bad
    return pd.Series(np.where(bad, "bad", np.where(good, "good", "average")), index=w.index)


def trend_reference(district: str, year: int) -> float:
    """Linear trend of APY in year (leave-one-year-out when ``year`` has an APY value)."""
    a = load_apy()
    a = a[(a.district == district) & (a.year != year)]
    return float(np.polyval(np.polyfit(a.year, a.yield_t_ha, 1), year))


def build_scenarios() -> list[YieldPrediction]:
    """Return the 30 analog-year scenarios."""
    w = pd.read_parquet(PROCESSED / "weather_district_year.parquet")
    w["scenario"] = classify(w)
    mean_rain = w.groupby("district").kharif_rainfall_mm.transform("mean")
    w["rain_vs_mean"] = (w.kharif_rainfall_mm / mean_rain - 1) * 100
    preds = district_year_predictions().set_index(["district", "year"]).pred
    out = []
    for r in (
        SimpleNamespace(**{str(k): v for k, v in d.items()}) for d in w.to_dict("records")
    ):  # attribute access, typed Any
        onset = date(int(r.year), 1, 1) + timedelta(days=int(r.monsoon_onset_day) - 1)
        out.append(
            YieldPrediction(
                district=str(r.district).replace("Kanpur", "Kanpur "),
                scenario=str(r.scenario),
                analog_year=int(r.year),
                planting_date=f"1 July {int(r.year)}",
                predicted_yield_t_ha=float(preds[(r.district, r.year)]),
                trend_reference_t_ha=trend_reference(str(r.district), int(r.year)),
                weather={
                    "kharif_rainfall_mm": round(float(r.kharif_rainfall_mm)),
                    "rain_vs_mean": f"{r.rain_vs_mean:+.0f}%",
                    "monsoon_onset": onset.strftime("%d %B"),
                    "dry_spells": int(r.dry_spell_count),
                    "heat_stress_days": int(r.heat_stress_days),
                },
            )
        )
    return out
