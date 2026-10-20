"""Assemble the Phase 3 report tables (sections A-H) from models/results/*.json."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.targets import load_apy
from src.models import common as c
from src.utils.seed import GLOBAL_SEED

R = c.RESULTS
PIXEL_MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "bayesian_ridge",
    "random_forest",
    "gbr",
    "xgboost",
    "lightgbm",
    "pymc_hierarchical",
]
VARIANTS = [
    "xgboost",
    "xgb_no_raw_peak",
    "xgb_no_satellite",
    "xgb_with_aquacrop",
    "xgb_no_raw_peak_with_aquacrop",
]


def load(name: str) -> dict[str, Any] | None:
    """Load the result JSON for ``name`` (None if absent)."""
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    """Markdown table."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append(
            "| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |"
        )
    return "\n".join(out)


def pixel_table(names: list[str]) -> pd.DataFrame:
    """Section A/B: test R2, RMSE, MAE combined and per district."""
    rows = []
    for n in names:
        r = load(n)
        if not r:
            continue
        t = r["test"]
        row: dict[str, Any] = {"model": n}
        for k, lab in (("combined", "all"), ("KanpurNagar", "Nagar"), ("KanpurDehat", "Dehat")):
            row[f"R2 {lab}"] = t[k]["r2"]
            row[f"RMSE {lab}"] = t[k]["rmse"]
            row[f"MAE {lab}"] = t[k]["mae"]
        row["CV R2"] = r.get("cv_r2", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def agg_rows(names: list[str], key: str) -> pd.DataFrame:
    """District-aggregated vs APY (pooled and per district) for result key ``key``."""
    rows = []
    for n in names:
        r = load(n)
        if not r or key not in r or "pooled" not in r[key]:
            continue
        a = r[key]
        row: dict[str, Any] = {
            "model": n,
            "N": a["n"],
            "pooled r": a["pooled"].get("pearson_r", float("nan")),
            "pooled R2": a["pooled"]["r2"],
            "pooled RMSE": a["pooled"]["rmse"],
        }
        for d, lab in (("KanpurNagar", "Nagar"), ("KanpurDehat", "Dehat")):
            if d in a:
                row[f"{lab} r"] = a[d]["pearson_r"]
                row[f"{lab} R2"] = a[d]["r2"]
                row[f"{lab} RMSE"] = a[d]["rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def climatology() -> dict[str, Any]:
    """Leave-one-year-out 'predict the district mean of the other years' reference."""
    apy = load_apy()
    out = {}
    pred = apy.copy()
    pred["p"] = [
        apy[(apy.district == d) & (apy.year != y)].yield_t_ha.mean()
        for d, y in zip(apy.district, apy.year, strict=True)
    ]
    out["pooled"] = c.metrics(pred.yield_t_ha.to_numpy(), pred.p.to_numpy())
    for d in c.DISTRICTS:
        s = pred[pred.district == d]
        out[d] = c.metrics(s.yield_t_ha.to_numpy(), s.p.to_numpy())
    return out


def trend_reference() -> dict[str, Any]:
    """Leave-one-year-out per-district linear trend in year (APY has a strong upward trend)."""
    apy = load_apy()
    pred = apy.copy()
    pred["p"] = [
        float(
            np.polyval(
                np.polyfit(
                    apy[(apy.district == d) & (apy.year != y)].year,
                    apy[(apy.district == d) & (apy.year != y)].yield_t_ha,
                    1,
                ),
                y,
            )
        )
        for d, y in zip(apy.district, apy.year, strict=True)
    ]
    out: dict[str, Any] = {"pooled": c.metrics(pred.yield_t_ha.to_numpy(), pred.p.to_numpy())}
    out["pooled_r"] = float(np.corrcoef(pred.yield_t_ha, pred.p)[0, 1])
    for d in c.DISTRICTS:
        s = pred[pred.district == d]
        out[d] = c.metrics(s.yield_t_ha.to_numpy(), s.p.to_numpy())
    return out


def residuals(name: str = "xgboost") -> str:
    """Section H: residual diagnostics and figure."""
    df = pd.read_parquet(c.PREDS / f"{name}.parquet")
    df["res"] = df.y - df.pred
    lines = [f"Overall mean residual (target - prediction): {df.res.mean():+.4f} t/ha"]
    by_d = df.groupby("district").res.agg(["mean", "std"]).round(4)
    by_y = df.groupby(["year", "district"]).res.mean().unstack().round(4)
    df["decile"] = pd.qcut(df.pred, 10, labels=False, duplicates="drop")
    by_q = df.groupby("decile").res.mean().round(4)
    lines += ["", "By district:", by_d.to_string(), "", "By year x district:", by_y.to_string()]
    lines += ["", "Mean residual by prediction decile (0=lowest):", by_q.to_string()]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    s = df.sample(30_000, random_state=GLOBAL_SEED)
    ax[0].scatter(s.pred, s.res, s=2, alpha=0.2)
    ax[0].axhline(0, color="k")
    ax[0].set(xlabel="predicted (t/ha)", ylabel="residual", title=f"{name}: residual vs predicted")
    for d, g in df.groupby("district"):
        ax[1].hist(g.res, bins=80, alpha=0.5, label=d, density=True)
    ax[1].legend()
    ax[1].set(title="Residuals by district", xlabel="residual (t/ha)")
    yrs = sorted(df.year.unique())
    ax[2].boxplot([df[df.year == y].res for y in yrs], tick_labels=yrs, showfliers=False)
    ax[2].axhline(0, color="k")
    ax[2].set(title="Residuals by test year", xlabel="year")
    fig.tight_layout()
    fig.savefig("figures/residuals_xgboost.png", dpi=130)
    return "\n".join(lines)


def main() -> None:
    """Write models/results/phase3_report.md."""
    parts: list[str] = []
    parts.append("## Clipping (ADR-029)\n" + md(pd.read_csv(R / "clip_stats.csv")))
    parts.append("## A. Pixel-level test R2 (2021-2024)\n" + md(pixel_table(PIXEL_MODELS)))
    parts.append("## B. raw_peak_ndvi sensitivity (XGBoost)\n" + md(pixel_table(VARIANTS)))
    parts.append("## C. District-aggregated vs APY: leave-one-year-out (HEADLINE)")
    parts.append(md(agg_rows(["linear", "ridge", "xgboost", *VARIANTS[1:]], "agg_apy_loyo")))
    clim = climatology()
    parts.append(
        "Reference (predict the district mean of the other APY years): "
        f"pooled R2 {clim['pooled']['r2']:.3f}, RMSE {clim['pooled']['rmse']:.3f}; "
        f"Nagar R2 {clim['KanpurNagar']['r2']:.3f}, Dehat R2 {clim['KanpurDehat']['r2']:.3f}"
    )
    tr = trend_reference()
    parts.append(
        "Reference (per-district linear time trend, leave-one-year-out): "
        f"pooled R2 {tr['pooled']['r2']:.3f}, RMSE {tr['pooled']['rmse']:.3f}, "
        f"r {tr['pooled_r']:.3f}; "
        f"Nagar R2 {tr['KanpurNagar']['r2']:.3f}, Dehat R2 {tr['KanpurDehat']['r2']:.3f}"
    )
    parts.append("### C2. All 13 APY years, in-sample for 2010-2020 (not a test)")
    parts.append(md(agg_rows(PIXEL_MODELS + VARIANTS[1:], "agg_apy_all_years")))
    parts.append("### C3. Out-of-sample years only (2021-2022, N=4)")
    parts.append(md(agg_rows(["ridge", "xgboost", *VARIANTS[1:]], "agg_apy_test_years")))
    rows = []
    for n in [
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "xgboost",
        "lightgbm",
        "random_forest",
        "gbr",
    ]:
        r = load(n)
        if r and "lodo" in r:
            for k, v in r["lodo"].items():
                rows.append({"model": n, "split": k, **v})
    parts.append("## Leave-one-district-out (pixel level)\n" + md(pd.DataFrame(rows)))
    imp = (
        pd.read_csv(R / "shap_importance.csv").head(10)
        if (R / "shap_importance.csv").exists()
        else None
    )
    if imp is not None:
        parts.append(
            "## E. SHAP top-10 (XGBoost, 5,000 test pixel-years)\n"
            + md(imp.rename(columns={"Unnamed: 0": "feature"}), "{:.4f}")
        )
    if (R / "bayes_hdi_20_pixels.csv").exists():
        parts.append(
            "## F. Bayesian 95% HDI widths, 20 random test pixels\n"
            + md(pd.read_csv(R / "bayes_hdi_20_pixels.csv"))
        )
        b = load("pymc_hierarchical")
        if b:
            parts.append(
                json.dumps(
                    {
                        k: b[k]
                        for k in (
                            "sampler",
                            "coverage_95_predictive_test",
                            "coverage_95_mean_only_test",
                            "mean_ppc_width_test",
                        )
                        if k in b
                    },
                    indent=1,
                )
            )
    aq = load("aquacrop")
    if aq:
        parts.append("## G. AquaCrop standalone\n```\n" + json.dumps(aq, indent=1) + "\n```")
    parts.append("## H. Residuals (XGBoost)\n```\n" + residuals() + "\n```")
    (R / "phase3_report.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
