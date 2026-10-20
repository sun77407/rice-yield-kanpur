"""ADR-031 sensitivity: XGBoost with raw_peak_ndvi dropped, no satellite, and AquaCrop feature.

All variants reuse the hyper-parameters tuned for the full feature set (no re-tuning), so the
comparison isolates the feature change; a re-tuned dropped-feature model could score slightly higher.
"""

from __future__ import annotations

import json

import pandas as pd

from src.models import common as c
from src.models.evaluate import evaluate
from src.models.feature_sets import FULL, NO_RAW_PEAK, NO_SATELLITE
from src.models.run_trees import build

AQ = "aquacrop_sim_yield"


def main() -> None:
    """Run the XGBoost variants with full protocol (LODO + leave-one-year-out APY)."""
    x = c.load_matrix()
    aq = pd.read_csv(c.PROCESSED / "aquacrop_district_year.csv")
    x = x.merge(aq, on=["district", "year"], how="left")
    assert x[AQ].notna().all()
    params = json.loads((c.RESULTS / "xgboost.json").read_text())["best_params"]
    variants = {
        "xgb_no_raw_peak": NO_RAW_PEAK,
        "xgb_no_satellite": NO_SATELLITE,
        "xgb_with_aquacrop": [*FULL, AQ],
        "xgb_no_raw_peak_with_aquacrop": [*NO_RAW_PEAK, AQ],
    }
    for name, feats in variants.items():
        evaluate(
            name,
            lambda: build("xgboost", params),
            feats,
            x,
            do_loyo=True,
            extra={"best_params_reused_from": "xgboost"},
        )


if __name__ == "__main__":
    main()
