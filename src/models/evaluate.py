"""Common evaluation protocol: spatiotemporal test, leave-one-district-out, district aggregation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.models import common as c

Factory = Callable[[], BaseEstimator]


def _fit_predict(
    make: Factory, feats: list[str], tr: pd.DataFrame, te: pd.DataFrame
) -> tuple[BaseEstimator, np.ndarray]:
    est = clone(make())
    est.fit(tr[feats], tr["y"])  # type: ignore[attr-defined]
    return est, np.asarray(est.predict(te[feats]))  # type: ignore[attr-defined]


def lodo(make: Factory, feats: list[str], x: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-district-out in two variants, both directions.

    ``spatial``: train on district A in every year, test on district B in every year (weather
    years are shared by both districts, so this tests spatial transfer only).
    ``spatiotemporal``: train on A in 2010-2020, test on B in 2021-2024.
    The district dummy is dropped because it is constant within a training district.
    """
    f = [col for col in feats if col != "district_kanpur_dehat"]
    out: dict[str, Any] = {}
    for a, b in (("KanpurNagar", "KanpurDehat"), ("KanpurDehat", "KanpurNagar")):
        for variant in ("spatial", "spatiotemporal"):
            if variant == "spatial":
                tr, te = x[x.district == a], x[x.district == b]
            else:
                tr = x[(x.district == a) & x.year.isin(c.TRAIN_YEARS)]
                te = x[(x.district == b) & x.year.isin(c.TEST_YEARS)]
            _, p = _fit_predict(make, f, tr, te)
            out[f"train_{a}_test_{b}_{variant}"] = c.metrics(te["y"].to_numpy(), p)
    return out


def loyo_district_agg(make: Factory, feats: list[str], x: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-year-out: predict each APY year from a model that never saw that year.

    This is the only district-level comparison with APY that is not in-sample.
    """
    apy_years = sorted(c.load_apy().year.unique())
    parts = []
    for yr in apy_years:
        tr, te = x[x.year != yr], x[x.year == yr]
        _, p = _fit_predict(make, feats, tr, te)
        t = te[["district", "year"]].copy()
        t["pred"] = p
        parts.append(t)
    return c.district_agg(pd.concat(parts).assign(y=0.0), "pred")


def evaluate(
    name: str,
    make: Factory,
    feats: list[str],
    x: pd.DataFrame,
    *,
    do_lodo: bool = True,
    do_loyo: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on 2010-2020, test on 2021-2024, and run the secondary/tertiary checks."""
    c.check_features(feats)
    tr, te = c.split(x)
    t0 = time.time()
    est, p = _fit_predict(make, feats, tr, te)
    fit_s = time.time() - t0
    te = te.assign(pred=p)
    ptr = np.asarray(est.predict(tr[feats]))  # type: ignore[attr-defined]
    full = x.assign(pred=np.asarray(est.predict(x[feats])))  # type: ignore[attr-defined]
    res: dict[str, Any] = {
        "model": name,
        "n_features": len(feats),
        "features": feats,
        "fit_seconds": round(fit_s, 1),
        "train": c.metrics(tr["y"].to_numpy(), ptr),
        "test": c.metrics_by_district(te),
        # all 13 APY years: 2010-2020 are in-sample, 2021-2022 are out-of-sample
        "agg_apy_all_years": c.district_agg(full),
        "agg_apy_test_years": c.district_agg(te),
        "bias_test_mean_residual": float((te["y"] - te["pred"]).mean()),
        **(extra or {}),
    }
    if do_lodo:
        res["lodo"] = lodo(make, feats, x)
    if do_loyo:
        res["agg_apy_loyo"] = loyo_district_agg(make, feats, x)
    c.save_preds(name, te, p)
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    (c.RESULTS / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = res["test"]["combined"]
    print(f"{name} test R2 {t['r2']:.3f} RMSE {t['rmse']:.3f} MAE {t['mae']:.3f}", flush=True)
    return res
