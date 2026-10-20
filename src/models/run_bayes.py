"""Agent 3C: Bayesian Ridge baseline and a PyMC hierarchical model.

NUTS on ~1M rows is not tractable here (no C compiler, pytensor runs in numba mode), so the
hierarchical model is fitted on a district-year-stratified training subsample and predicts every
test pixel from the posterior. Features are median-imputed and standardised with training-subsample
statistics; censoring indicators are added for the three nullable phenology columns.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.linear_model import BayesianRidge

from src.models import common as c
from src.models.evaluate import evaluate
from src.models.feature_sets import FULL
from src.models.run_baselines import pipe

SEED = c.GLOBAL_SEED
N_TRAIN = 30_000
QUICK = os.environ.get("BAYES_QUICK") == "1"  # smoke test of the post-processing path
DRAWS, TUNE = (60, 60) if QUICK else (2000, 1000)
FEATS = [f for f in FULL if f != "district_kanpur_dehat"]  # district handled by the hierarchy
NULLABLE = ["greenup_day", "senescence_day", "length_of_season"]


def hdi(draws: np.ndarray, prob: float = 0.95) -> np.ndarray:
    """Shortest interval containing ``prob`` of the draws (highest density interval)."""
    d = np.sort(draws)
    k = int(np.ceil(prob * len(d)))
    w = d[k - 1 :] - d[: len(d) - k + 1]
    i = int(np.argmin(w))
    return np.array([d[i], d[i + k - 1]])


def design(df: pd.DataFrame, med: pd.Series, mu: pd.Series, sd: pd.Series) -> np.ndarray:
    """Impute with training medians, standardise, append censoring indicators."""
    ind = df[NULLABLE].isna().astype(float).to_numpy()
    z = ((df[FEATS].fillna(med) - mu) / sd).to_numpy()
    return np.hstack([z, ind])


def main() -> None:
    """Fit Bayesian Ridge and the PyMC model, then report posterior intervals."""
    x = c.load_matrix()
    tr, te = c.split(x)
    evaluate("bayesian_ridge", lambda: pipe(BayesianRidge()), FULL, x, do_loyo=False)

    sub = (
        tr.groupby(["district", "year"])
        .sample(n=N_TRAIN // 22, random_state=SEED)
        .reset_index(drop=True)
    )
    med = sub[FEATS].median()
    mu = sub[FEATS].fillna(med).mean()
    sd = sub[FEATS].fillna(med).std().replace(0, 1)
    xs = design(sub, med, mu, sd)
    names = [*FEATS, *[f"missing_{n}" for n in NULLABLE]]
    d_idx = (sub.district == "KanpurNagar").astype(int).to_numpy()
    y = sub["y"].to_numpy()
    coords = {"feature": names, "district": ["KanpurDehat", "KanpurNagar"]}
    with pm.Model(coords=coords):
        mu0 = pm.Normal("mu0", float(y.mean()), 1.0)
        tau = pm.HalfNormal("tau", 0.5)
        mu_d = pm.Normal("mu_district", mu0, tau, dims="district")
        beta = pm.Normal("beta", 0.0, 0.5, dims="feature")
        sigma = pm.HalfNormal("sigma_pixel", 1.0)
        pm.Normal("y", mu_d[d_idx] + pm.math.dot(xs, beta), sigma, observed=y)
        t0 = time.time()
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=2,
            cores=2,
            random_seed=SEED,
            compile_kwargs={"mode": "NUMBA"},
            progressbar=False,
        )
    sample_s = time.time() - t0
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf("models/bayes_trace.nc")  # arviz 1.x: DataTree method
    summ = az.summary(idata, var_names=["mu0", "tau", "mu_district", "beta", "sigma_pixel"])
    summ.to_csv(c.RESULTS / "bayes_summary.csv")
    div = int(idata.sample_stats["diverging"].sum())
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    print(f"NUTS {sample_s:.0f}s divergences={div} max_rhat={rhat_max:.3f} min_ess={ess_min:.0f}")

    post = idata["posterior"].to_dataset().stack(s=("chain", "draw"))
    b = post["beta"].to_numpy()  # (feature, s)
    m_d = post["mu_district"].to_numpy()  # (district, s)
    sig = post["sigma_pixel"].to_numpy()
    xt = design(te, med, mu, sd)
    dt = (te.district == "KanpurNagar").astype(int).to_numpy()
    pm_mean = m_d.mean(axis=1)[dt] + xt @ b.mean(axis=1)
    tt = te.assign(pred=pm_mean)
    c.save_preds("pymc_hierarchical", tt, pm_mean)
    full = x.assign(
        pred=m_d.mean(axis=1)[(x.district == "KanpurNagar").astype(int).to_numpy()]
        + design(x, med, mu, sd) @ b.mean(axis=1)
    )
    res: dict[str, Any] = {
        "model": "pymc_hierarchical",
        "train_rows": len(sub),
        "test": c.metrics_by_district(tt),
        "agg_apy_all_years": c.district_agg(full),
        "agg_apy_test_years": c.district_agg(tt),
        "sampler": {
            "seconds": sample_s,
            "divergences": div,
            "max_rhat": rhat_max,
            "min_ess_bulk": ess_min,
        },
    }

    rng = np.random.default_rng(SEED)
    pick = te.sample(20_000, random_state=SEED).index
    ts = te.loc[pick]
    xs_t = design(ts, med, mu, sd)
    dts = (ts.district == "KanpurNagar").astype(int).to_numpy()
    draws = rng.choice(b.shape[1], min(500, b.shape[1]), replace=False)
    mean_draws = m_d[dts][:, draws] + xs_t @ b[:, draws]  # (n, 500) epistemic only
    ppc = mean_draws + rng.standard_normal(mean_draws.shape) * sig[draws]
    lo_m, hi_m = np.percentile(mean_draws, [2.5, 97.5], axis=1)
    lo_p, hi_p = np.percentile(ppc, [2.5, 97.5], axis=1)
    hdi_m = np.stack([hdi(mean_draws[i]) for i in range(20)])
    hdi_p = np.stack([hdi(ppc[i]) for i in range(20)])
    yv = ts["y"].to_numpy()
    res["coverage_95_predictive_test"] = float(((yv >= lo_p) & (yv <= hi_p)).mean())
    res["coverage_95_mean_only_test"] = float(((yv >= lo_m) & (yv <= hi_m)).mean())
    res["mean_ppc_width_test"] = float((hi_p - lo_p).mean())
    sample20 = pd.DataFrame(
        {
            "pixel_id": ts["pixel_id"].to_numpy()[:20],
            "district": ts["district"].to_numpy()[:20],
            "year": ts["year"].to_numpy()[:20],
            "y_true": yv[:20],
            "pred_mean": mean_draws[:20].mean(axis=1),
            "hdi95_mean_width": hdi_m[:, 1] - hdi_m[:, 0],
            "hdi95_predictive_width": hdi_p[:, 1] - hdi_p[:, 0],
        }
    )
    sample20.to_csv(c.RESULTS / "bayes_hdi_20_pixels.csv", index=False)
    (c.RESULTS / "pymc_hierarchical.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    k = 300
    ax[0].errorbar(
        yv[:k],
        ppc[:k].mean(axis=1),
        yerr=[ppc[:k].mean(axis=1) - lo_p[:k], hi_p[:k] - ppc[:k].mean(axis=1)],
        fmt="o",
        ms=2,
        alpha=0.4,
    )
    ax[0].plot([0.5, 5], [0.5, 5], "k--")
    ax[0].set(
        xlabel="target (t/ha)",
        ylabel="posterior predictive mean",
        title="Test pixels, 95% predictive interval",
    )
    ax[1].hist(yv, bins=60, density=True, alpha=0.5, label="observed target")
    ax[1].hist(ppc[:, 0], bins=60, density=True, alpha=0.5, label="posterior predictive (1 draw)")
    ax[1].legend()
    ax[1].set(title="Posterior predictive check (test)", xlabel="t/ha")
    bm, bs = b.mean(axis=1), b.std(axis=1)
    ax[2].errorbar(bm, range(len(names)), xerr=1.96 * bs, fmt="o", ms=3)
    ax[2].set_yticks(range(len(names)), names, fontsize=6)
    ax[2].set_title("beta posteriors (standardised)")
    fig.tight_layout()
    fig.savefig("figures/bayes_posterior_predictive.png", dpi=130)
    t = res["test"]["combined"]
    print(
        f"pymc test R2 {t['r2']:.3f} RMSE {t['rmse']:.3f}; coverage {res['coverage_95_predictive_test']:.3f}"
    )
    print(sample20.round(3).to_string())


if __name__ == "__main__":
    main()
