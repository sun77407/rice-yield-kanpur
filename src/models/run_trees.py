"""Agent 3B: RF, sklearn GBR, XGBoost, LightGBM with Optuna (5-fold CV inside train only).

CV folds are grouped by year (weather is constant within a district-year, so random folds
would leak). Tuning runs on a random subsample of the training rows; final fits use all training
rows except sklearn GBR (subsampled, it is too slow on ~1M rows). The test split (2021-2024) is
never touched during tuning or model selection: the best model is chosen by CV R2.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from src.models import common as c
from src.models.evaluate import evaluate
from src.models.feature_sets import FULL
from src.utils.seed import GLOBAL_SEED

MODELS = Path("models")
TUNE_ROWS = {"xgboost": 150_000, "lightgbm": 150_000, "random_forest": 80_000, "gbr": 40_000}
TRIALS = {"xgboost": 30, "lightgbm": 30, "random_forest": 12, "gbr": 10}
TIMEOUT_S = 900
GBR_FIT_ROWS = 300_000
SEED = GLOBAL_SEED


def xgb_device() -> str:
    """Use the GPU when XGBoost can run on it, else CPU."""
    try:
        m = xgb.XGBRegressor(n_estimators=2, device="cuda", tree_method="hist")
        m.fit(np.random.rand(200, 3), np.random.rand(200))
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = xgb_device()


def space_xgb(t: optuna.Trial) -> dict[str, Any]:
    """XGBoost search space."""
    return {
        "n_estimators": t.suggest_int("n_estimators", 150, 600),
        "max_depth": t.suggest_int("max_depth", 3, 9),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 100, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True),
    }


def space_lgb(t: optuna.Trial) -> dict[str, Any]:
    """LightGBM search space."""
    return {
        "n_estimators": t.suggest_int("n_estimators", 150, 600),
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True),
    }


def space_rf(t: optuna.Trial) -> dict[str, Any]:
    """Random forest search space."""
    return {
        "max_depth": t.suggest_int("max_depth", 6, 30),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    }


def space_gbr(t: optuna.Trial) -> dict[str, Any]:
    """Search space for the sklearn GBR."""
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 300),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 300, log=True),
    }


def build(name: str, p: dict[str, Any], tuning: bool = False) -> Any:
    """Estimator factory for ``name`` with parameters ``p``."""
    if name == "xgboost":
        return xgb.XGBRegressor(
            **p, tree_method="hist", device=DEVICE, random_state=SEED, n_jobs=16
        )
    if name == "lightgbm":
        return lgb.LGBMRegressor(**p, random_state=SEED, n_jobs=16, verbosity=-1)
    if name == "random_forest":
        return RandomForestRegressor(
            **p,
            n_estimators=60 if tuning else 250,
            max_samples=0.3,
            n_jobs=16,
            random_state=SEED,
        )
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        GradientBoostingRegressor(**p, random_state=SEED),
    )


SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "xgboost": space_xgb,
    "lightgbm": space_lgb,
    "random_forest": space_rf,
    "gbr": space_gbr,
}


def tune(name: str, tr: pd.DataFrame) -> tuple[dict[str, Any], float, int]:
    """Optuna TPE search; returns (best params, best CV R2, completed trials)."""
    sub = tr.sample(min(TUNE_ROWS[name], len(tr)), random_state=SEED)
    x, y, g = sub[FULL], sub["y"].to_numpy(), sub["year"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, g))

    def objective(t: optuna.Trial) -> float:
        p = SPACES[name](t)
        scores = []
        for a, b in folds:
            m = build(name, p, tuning=True)
            m.fit(x.iloc[a], y[a])
            pr = m.predict(x.iloc[b])
            scores.append(1 - ((y[b] - pr) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum())
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS[name], timeout=TIMEOUT_S)
    return study.best_params, float(study.best_value), len(study.trials)


def main() -> None:
    """Tune, evaluate, then export the CV-best model and SHAP for XGBoost."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x = c.load_matrix()
    tr, _ = c.split(x)
    summary: dict[str, Any] = {"xgb_device": DEVICE}
    fitted: dict[str, Any] = {}
    for name in ("xgboost", "lightgbm", "random_forest", "gbr"):
        t0 = time.time()
        params, cv, n = tune(name, tr)
        print(
            name,
            f"CV R2 {cv:.4f} after {n} trials ({time.time() - t0:.0f}s)",
            params,
            flush=True,
        )

        def make(name: str = name, params: dict[str, Any] = params) -> Any:
            return build(name, params)

        fit_tr = tr.sample(GBR_FIT_ROWS, random_state=SEED) if name == "gbr" else tr
        full_x = x if name != "gbr" else pd.concat([fit_tr, x[~x.index.isin(tr.index)]])
        res = evaluate(
            name,
            make,
            FULL,
            full_x,
            do_loyo=name == "xgboost",
            extra={
                "cv_r2": cv,
                "trials": n,
                "best_params": params,
                "fit_rows": len(fit_tr),
                "tuning_rows": TUNE_ROWS[name],
            },
        )
        summary[name] = {"cv_r2": cv, "test_r2": res["test"]["combined"]["r2"], "params": params}
        fitted[name] = make().fit(fit_tr[FULL], fit_tr["y"])
    best = max(("xgboost", "lightgbm", "random_forest", "gbr"), key=lambda k: summary[k]["cv_r2"])
    summary["best_by_cv"] = best
    MODELS.mkdir(exist_ok=True)
    with (MODELS / f"best_{best}.pkl").open("wb") as fh:
        pickle.dump(fitted[best], fh)
    with (MODELS / "xgboost.pkl").open("wb") as fh:
        pickle.dump(fitted["xgboost"], fh)
    summary["onnx"] = export_onnx(best, fitted[best], x)
    (c.RESULTS / "trees_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "onnx"}, indent=1)[:1500])
    print("onnx:", summary["onnx"])


def export_onnx(name: str, model: Any, x: pd.DataFrame) -> dict[str, Any]:
    """Export to ONNX and check onnxruntime reproduces the sklearn/xgboost predictions."""
    import onnxruntime as ort
    from onnxmltools.convert.common.data_types import FloatTensorType

    n = len(FULL)
    out = MODELS / f"best_{name}.onnx"
    sample = x[FULL].sample(2000, random_state=SEED).astype("float32")
    try:
        spec = [("input", FloatTensorType([None, n]))]
        if name == "xgboost":
            import onnxmltools

            m = pickle.loads(pickle.dumps(model))
            m.get_booster().feature_names = None
            onx = onnxmltools.convert_xgboost(m, initial_types=spec, target_opset=15)
        elif name == "lightgbm":
            import onnxmltools

            onx = onnxmltools.convert_lightgbm(model, initial_types=spec, target_opset=15)
        else:
            from skl2onnx import convert_sklearn

            onx = convert_sklearn(model, initial_types=spec, target_opset=15)
        out.write_bytes(onx.SerializeToString())
        sess = ort.InferenceSession(str(out))
        pred = sess.run(None, {"input": sample.to_numpy()})[0].ravel()
        ref = np.asarray(model.predict(sample))
        return {"path": str(out), "max_abs_diff": float(np.abs(pred - ref).max()), "ok": True}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)[:300]}


if __name__ == "__main__":
    main()
