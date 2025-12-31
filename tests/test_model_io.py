from pathlib import Path
import joblib

def test_model_artifact_exists():
    assert Path("artifacts/model.joblib").exists()

def test_model_loads():
    m = joblib.load("artifacts/model.joblib")
    assert hasattr(m, "predict_proba")
