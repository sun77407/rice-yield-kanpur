"""Export the XGBoost model to ONNX and verify it against the native predictions.

onnxmltools reads ``Booster.get_dump(with_stats=True)`` as JSON; with XGBoost 3.x that dump
contains bare ``-inf`` statistics tokens (loss changes of unsplit nodes), which are not valid JSON
and which the converter does not use. They are replaced by 0 for the conversion only; the model
itself is untouched.
"""

from __future__ import annotations

import json
import pickle
import re
from typing import Any

import numpy as np
import onnxmltools
import onnxmltools.convert.xgboost.operator_converters.XGBoost as xconv
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from src.models import common as c
from src.models.feature_sets import FULL


class _TolerantJson:
    """``json`` stand-in whose ``loads`` accepts bare inf/nan tokens."""

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)

    def loads(self, text: str, *a: Any, **k: Any) -> Any:
        return json.loads(re.sub(r"-?\binf\b|\bnan\b", "0", text), *a, **k)


def main() -> None:
    """Convert models/xgboost.pkl to models/best_xgboost.onnx and check agreement."""
    with open("models/xgboost.pkl", "rb") as fh:
        model = pickle.load(fh)
    model.get_booster().feature_names = None
    xconv.json = _TolerantJson()  # type: ignore[attr-defined]
    onx = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, len(FULL)]))], target_opset=15
    )
    path = "models/best_xgboost.onnx"
    with open(path, "wb") as fh:
        fh.write(onx.SerializeToString())
    x = c.load_matrix().sample(5000, random_state=c.GLOBAL_SEED)[FULL].astype("float32")
    pred = ort.InferenceSession(path).run(None, {"input": x.to_numpy()})[0].ravel()
    ref = model.predict(x)
    diff = float(np.abs(pred - ref).max())
    print(f"ONNX written to {path}; max abs diff vs xgboost on 5,000 rows: {diff:.2e}")
    (c.RESULTS / "onnx_check.json").write_text(json.dumps({"path": path, "max_abs_diff": diff}))


if __name__ == "__main__":
    main()
