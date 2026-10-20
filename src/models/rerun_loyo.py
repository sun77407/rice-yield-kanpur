"""Refresh the leave-one-year-out APY aggregation (adds pooled Pearson r) for key models."""

from __future__ import annotations

import json

from sklearn.linear_model import LinearRegression, Ridge

from src.models import common as c
from src.models.evaluate import loyo_district_agg
from src.models.feature_sets import FULL, LINEAR
from src.models.run_baselines import pipe
from src.models.run_trees import build


def main() -> None:
    """Recompute and store ``agg_apy_loyo`` for linear, ridge, xgboost and lightgbm."""
    x = c.load_matrix()
    specs = {
        "linear": (lambda: pipe(LinearRegression()), LINEAR),
        "ridge": (lambda: pipe(Ridge(alpha=1.0)), FULL),
    }
    for name in ("xgboost", "lightgbm"):
        params = json.loads((c.RESULTS / f"{name}.json").read_text())["best_params"]
        specs[name] = (lambda n=name, p=params: build(n, p), FULL)  # type: ignore[misc]
    for name, (make, feats) in specs.items():
        p = c.RESULTS / f"{name}.json"
        res = json.loads(p.read_text())
        res["agg_apy_loyo"] = loyo_district_agg(make, feats, x)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(name, res["agg_apy_loyo"]["pooled"], flush=True)


if __name__ == "__main__":
    main()
