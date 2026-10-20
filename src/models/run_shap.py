"""SHAP for the XGBoost model (summary plot, top-5 dependence plots, top-10 table)."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import common as c
from src.models.feature_sets import FULL

FIG = Path("figures")


def main() -> None:
    """Compute SHAP on 5,000 random test pixel-years."""
    x = c.load_matrix()
    _, te = c.split(x)
    sample = te.sample(5000, random_state=c.GLOBAL_SEED)[FULL]
    with Path("models/xgboost.pkl").open("rb") as fh:
        model = pickle.load(fh)
    ex = shap.TreeExplainer(model)
    sv = ex(sample)
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=FULL).sort_values(ascending=False)
    imp.to_csv(c.RESULTS / "shap_importance.csv", header=["mean_abs_shap"])
    print(imp.head(10).round(4).to_string())
    shap.summary_plot(sv.values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=130)
    plt.close()
    for f in imp.index[:5]:
        shap.dependence_plot(f, sv.values, sample, show=False)
        plt.tight_layout()
        plt.savefig(FIG / f"shap_dependence_{f}.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
