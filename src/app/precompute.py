"""Precompute the district-year tables the app reads (so the app never refits a model).

For the full XGBoost and the ``raw_peak_ndvi``-free variant (ADR-031/032): the district-year mean
prediction of a model fitted on 2010-2020 (the headline fit), and the leave-one-year-out
prediction for every APY year (out-of-sample). Also the weather scenario label and APY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.targets import load_apy
from src.llm.scenarios import classify
from src.models import common as c
from src.models.feature_sets import FULL, NO_RAW_PEAK
from src.models.run_trees import build

OUT = Path("data/processed/app_tables.parquet")
VARIANTS = {"full": FULL, "noraw": NO_RAW_PEAK}


def main() -> None:
    """Fit both variants (hyper-parameters reused from the tuned model) and write the table."""
    x = c.load_matrix()
    params = json.loads((c.RESULTS / "xgboost.json").read_text())["best_params"]
    base = x[["district", "year"]].drop_duplicates().sort_values(["district", "year"])
    out = base.reset_index(drop=True)
    apy_years = sorted(load_apy().year.unique())
    for tag, feats in VARIANTS.items():
        tr = x[x.year.isin(c.TRAIN_YEARS)]
        est = build("xgboost", params).fit(tr[feats], tr["y"])
        d = x[["district", "year"]].assign(p=est.predict(x[feats]))
        out = out.merge(
            d.groupby(["district", "year"], as_index=False).agg(**{f"pred_{tag}": ("p", "mean")}),
            on=["district", "year"],
        )
        loyo = []
        for yr in apy_years:
            m = build("xgboost", params).fit(x[x.year != yr][feats], x[x.year != yr]["y"])
            te = x[x.year == yr]
            loyo.append(te[["district", "year"]].assign(p=m.predict(te[feats])))
        lo = pd.concat(loyo).groupby(["district", "year"], as_index=False).p.mean()
        out = out.merge(
            lo.rename(columns={"p": f"loyo_{tag}"}), on=["district", "year"], how="left"
        )
        print(tag, "done", flush=True)
    w = pd.read_parquet(c.PROCESSED / "weather_district_year.parquet")
    w["scenario"] = classify(w)
    out = out.merge(w[["district", "year", "scenario"]], on=["district", "year"])
    apy = load_apy().rename(columns={"yield_t_ha": "apy"})[["district", "year", "apy"]]
    out = out.merge(apy, on=["district", "year"], how="left")
    out.to_parquet(OUT, index=False)
    print(out.groupby("scenario").size().to_dict(), out.shape)


if __name__ == "__main__":
    main()
