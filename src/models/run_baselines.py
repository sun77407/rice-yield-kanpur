"""Agent 3A: Linear, Ridge, Lasso, ElasticNet (fixed settings, no tuning)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models import common as c
from src.models.evaluate import evaluate
from src.models.feature_sets import FULL, LINEAR


def pipe(model: object) -> Pipeline:
    """Median imputation (+ missingness indicators: censoring is informative) then scaling."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model
    )  # type: ignore[arg-type]


def vif_table(x: pd.DataFrame, cols: list[str]) -> pd.Series:
    """VIF on the training pixels for ``cols`` (median-imputed)."""
    d = x[cols].fillna(x[cols].median())
    d = d.loc[:, d.std() > 0]
    inv = np.linalg.pinv(np.corrcoef(d.to_numpy(), rowvar=False))
    return pd.Series(np.diag(inv), index=d.columns).sort_values(ascending=False)


def main() -> None:
    """Run the four baselines and record the linear-model VIF."""
    x = c.load_matrix()
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    c.clip_stats(x).to_csv(c.RESULTS / "clip_stats.csv", index=False)
    tr, _ = c.split(x)
    # Linear/Lasso/ElasticNet use LINEAR (no gdd_kharif, ADR-028); Ridge uses the full set.
    # Fixed regularisation (PROJECT_FLOW 2.6: baselines are not tuned), features standardised.
    specs = {
        "linear": (lambda: pipe(LinearRegression()), LINEAR),
        "ridge": (lambda: pipe(Ridge(alpha=1.0)), FULL),
        "lasso": (lambda: pipe(Lasso(alpha=1e-3, max_iter=5000)), LINEAR),
        "elasticnet": (lambda: pipe(ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)), LINEAR),
    }
    for name, (make, feats) in specs.items():
        evaluate(name, make, feats, x, do_loyo=name in {"linear", "ridge"})
    v = {
        "linear_set": vif_table(tr, LINEAR).round(1).to_dict(),
        "full_set": vif_table(tr, FULL).round(1).to_dict(),
    }
    (c.RESULTS / "vif_baselines.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    print({k: dict(list(d.items())[:5]) for k, d in v.items()})


if __name__ == "__main__":
    main()
