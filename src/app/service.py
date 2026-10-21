"""App logic (no Streamlit imports): yield intervals, trend reference, advisory + judge pipeline.

Honest-display rules (ADR-032): the model estimate is always shown next to the per-district
linear time-trend reference, and the wording changes with whether the two agree.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.llm.advisor import Advisory, YieldPrediction, generate_advisory
from src.llm.judge import Faithfulness, judge_advisory
from src.llm.scenarios import build_scenarios

TABLES = Path("data/processed/app_tables.parquet")
HISTORY = Path("logs/faithfulness_history.jsonl")
DISTRICTS = {"Kanpur Nagar": "KanpurNagar", "Kanpur Dehat": "KanpurDehat"}
SCENARIOS = ("bad", "average", "good")
PLANTING_WINDOW = (
    date(2000, 6, 15),
    date(2000, 7, 31),
)  # month/day window for Kharif transplanting


@dataclass
class YieldResult:
    """Yield estimate for one district under a weather scenario."""

    district: str
    scenario: str
    variant: str  # full | noraw
    analog_years: list[int]
    representative_year: int
    p10: float
    p50: float
    p90: float
    trend: float
    trend_rmse: float
    apy_mean: float
    apy_sd: float
    area_ha: float
    in_sample_years: list[int]
    verdict: str  # agree | model_higher | model_lower
    message: str

    @property
    def total_tonnes(self) -> tuple[float, float, float]:
        """P10 / P50 / P90 production for the entered area."""
        return (self.p10 * self.area_ha, self.p50 * self.area_ha, self.p90 * self.area_ha)


@lru_cache(maxsize=1)
def tables() -> pd.DataFrame:
    """District-year predictions, scenario labels and APY (see ``src.app.precompute``)."""
    return pd.read_parquet(TABLES)


def _apy(district: str) -> pd.DataFrame:
    t = tables()
    return t[(t.district == district) & t.apy.notna()].sort_values("year")


def trend_at(district: str, year: int) -> tuple[float, float]:
    """Linear APY trend evaluated at ``year`` and its leave-one-out residual RMSE."""
    a = _apy(district)
    coef = np.polyfit(a.year, a.apy, 1)
    loo = []
    for yr in a.year:
        m = a[a.year != yr]
        loo.append(a[a.year == yr].apy.iloc[0] - np.polyval(np.polyfit(m.year, m.apy, 1), yr))
    return float(np.polyval(coef, year)), float(np.sqrt(np.mean(np.square(loo))))


def planting_note(planting: date) -> str | None:
    """Warning when the planting date is outside the 15 Jun - 31 Jul window used in training."""
    md = date(2000, planting.month, planting.day)
    if PLANTING_WINDOW[0] <= md <= PLANTING_WINDOW[1]:
        return None
    return (
        "This planting date is outside the 15 June - 31 July window the satellite phenology was "
        "built for. The yield model does not use the planting date at all (see Diagnostics), so "
        "the estimate does not change; treat the advisory with extra care."
    )


def predict(
    district_label: str, planting: date, area_ha: float, scenario: str, variant: str = "full"
) -> YieldResult:
    """P10/P50/P90 for ``scenario`` and the comparison with the time-trend reference.

    The estimate for each historical analog year of the scenario is the model's district mean;
    the interval adds the empirical leave-one-year-out APY residuals of the same model (13 per
    district), so it reflects out-of-sample error, not only the spread across analog years.
    """
    district = DISTRICTS[district_label]
    t = tables()
    col, lo_col = f"pred_{variant}", f"loyo_{variant}"
    rows = t[(t.district == district) & (t.scenario == scenario)].sort_values("year")
    apy = _apy(district)
    resid = (apy.apy - apy[lo_col]).to_numpy()
    preds = rows[col].to_numpy()
    samples = (preds[:, None] + resid[None, :]).ravel()
    p10, p50, p90 = (float(np.quantile(samples, q)) for q in (0.1, 0.5, 0.9))
    trend, trend_rmse = trend_at(district, planting.year)
    rep = int(rows.iloc[(rows[col] - np.median(preds)).abs().argmin()].year)
    diff = p50 - trend
    if abs(diff) <= trend_rmse:
        verdict = "agree"
        msg = (
            f"Within the baseline range for {district_label}: the model ({p50:.2f} t/ha) and the "
            f"time trend ({trend:.2f} t/ha) agree to within the trend's own error "
            f"(+/-{trend_rmse:.2f})."
        )
    else:
        verdict = "model_higher" if diff > 0 else "model_lower"
        msg = (
            f"The model ({p50:.2f} t/ha) and the time trend ({trend:.2f} t/ha) disagree by "
            f"{abs(diff):.2f} t/ha, more than the trend's error (+/-{trend_rmse:.2f}). On the "
            "headline leave-one-year-out test the trend was the better predictor, so weigh the "
            "trend at least as heavily as the model."
        )
    last = int(apy.year.max())
    if planting.year > last:
        msg += (
            f" The trend is extrapolated {planting.year - last} years past the last APY year "
            f"({last}); the model has no time term, so it cannot follow a rising trend and "
            "will sit below it whenever the planting year is later than the data."
        )
    if abs(p50 - apy.apy.mean()) <= 0.5 * apy.apy.std():
        msg += f" The estimate is also close to the 2010-2022 mean ({apy.apy.mean():.2f} t/ha)."
    in_sample = [int(y) for y in rows.year if y <= 2020]
    return YieldResult(
        district=district_label,
        scenario=scenario,
        variant=variant,
        analog_years=[int(y) for y in rows.year],
        representative_year=rep,
        p10=p10,
        p50=p50,
        p90=p90,
        trend=trend,
        trend_rmse=trend_rmse,
        apy_mean=float(apy.apy.mean()),
        apy_sd=float(apy.apy.std()),
        area_ha=area_ha,
        in_sample_years=in_sample,
        verdict=verdict,
        message=msg,
    )


@lru_cache(maxsize=1)
def _scenarios() -> dict[tuple[str, int], YieldPrediction]:
    return {(s.district.replace(" ", ""), s.analog_year): s for s in build_scenarios()}


@dataclass
class AdvisoryRun:
    """An advisory with its faithfulness verdict and timings."""

    advisory: Advisory
    faithfulness: Faithfulness
    total_s: float
    generate_s: float
    judge_s: float


def run_advisory(y: YieldResult, planting: date) -> AdvisoryRun:
    """Retrieve + write (qwen), then judge claim by claim against cited passages (llama3)."""
    t0 = time.time()
    base = _scenarios()[(DISTRICTS[y.district], y.representative_year)]
    pred = replace(
        base,
        planting_date=f"{planting.day} {planting.strftime('%B %Y')}",
        predicted_yield_t_ha=y.p50,
        trend_reference_t_ha=y.trend,
    )
    adv = generate_advisory(pred)
    t1 = time.time()
    texts = [f"{p.title}. {p.text}" for p in adv.passages]
    f = judge_advisory(adv.sentences, texts)
    t2 = time.time()
    run = AdvisoryRun(adv, f, t2 - t0, t1 - t0, t2 - t1)
    log_history(y, run)
    return run


def log_history(y: YieldResult, run: AdvisoryRun) -> None:
    """Append one faithfulness record (user-visible 'grounding score history')."""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "app",
        "district": y.district,
        "scenario": y.scenario,
        "variant": y.variant,
        "frac_supported": run.faithfulness.frac_supported,
        "frac_partial": run.faithfulness.frac_partial,
        "frac_unsupported": run.faithfulness.frac_unsupported,
        "cosine_relevance": run.advisory.cosine_relevance_score,
        "round_trip_s": round(run.total_s, 1),
    }
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def read_history() -> pd.DataFrame:
    """All logged faithfulness records."""
    if not HISTORY.exists():
        return pd.DataFrame()
    return pd.read_json(HISTORY, lines=True)
