# app/main.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_names.json"

app = FastAPI(title="Tabular ML Service", version="1.0.0")

_model = None
_features = None


def load_artifacts():
    global _model, _features
    if _model is None:
        if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
            raise RuntimeError("Missing artifacts. Run training first to create artifacts/model.joblib and feature_names.json")
        _model = joblib.load(MODEL_PATH)
        _features = json.loads(FEATURES_PATH.read_text())


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/schema")
def schema():
    load_artifacts()
    return {"required_features": _features}


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Payload example:
    {
      "mean radius": 14.2,
      "mean texture": 20.1,
      ...
    }
    """
    load_artifacts()

    missing = [f for f in _features if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Build row in correct feature order
    x = np.array([[float(payload[f]) for f in _features]], dtype=float)

    proba = float(_model.predict_proba(x)[:, 1][0])
    pred = int(proba >= 0.5)

    return {"prediction": pred, "probability_positive": proba}
