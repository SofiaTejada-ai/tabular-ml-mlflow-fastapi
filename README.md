# End-to-End Tabular ML System (LightGBM, Optuna, MLflow, SHAP, FastAPI, Streamlit)

This repo is a **mini real-world ML product pipeline**: it trains a tabular classifier, tunes it automatically, tracks experiments, generates explainability plots, and serves predictions through an API + simple UI.

**What you can do here**
- Train a strong tabular model (**LightGBM**)
- Auto-tune hyperparameters (**Optuna**) across many trials
- Track runs (metrics/params/artifacts) in a “W&B-like” dashboard (**MLflow**, local)
- Generate explainability visuals (**SHAP**)
- Serve predictions via **FastAPI** (`/predict`) and a simple **Streamlit** demo UI
- (Optional) Package the API with **Docker**

---

## Tech Stack
- Modeling: `lightgbm`, `scikit-learn`
- Tuning: `optuna`
- Tracking: `mlflow` (local `mlruns/`)
- Explainability: `shap`, `matplotlib`
- Serving: `fastapi`, `uvicorn`
- Demo UI: `streamlit`
- Utilities/testing: `pandas`, `numpy`, `joblib`, `pytest`

---

## Project Structure

src/
data.py # loads dataset
train.py # Optuna HPO + MLflow logging + saves best model artifacts
explain.py # SHAP plot generation
app/
main.py # FastAPI inference service (health/schema/predict)
ui/
streamlit_app.py # simple demo UI (Load example -> Predict)
artifacts/
model.joblib # created after training (NOT typically committed)
feature_names.json # created after training
best_params.json # created after training
shap/ # SHAP images + metadata (these are nice to commit)
mlruns/ # MLflow run storage (usually ignored)


---

## What This Demo Predicts
By default, this project uses a **built-in scikit-learn dataset** (so anyone can run it instantly without downloading data).  
The goal of this repo is to showcase the **end-to-end ML workflow**: **train → tune → track → explain → serve**.

> Want a more “business” dataset (churn/fraud/loan default)? You can swap `src/data.py` to load a CSV and keep the rest of the pipeline unchanged.

---

## Quickstart

### 1) Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Train + Tune + Log to MLflow

This runs Optuna trials, logs everything to MLflow, and creates model artifacts in artifacts/.
```
python -m src.train --trials 30
```

After training, you should have:

artifacts/model.joblib

artifacts/feature_names.json

artifacts/best_params.json

artifacts/shap/shap_summary_bar.png

artifacts/shap/shap_summary_beeswarm.png

## View Experiments in MLflow

Start the MLflow UI:
```
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open:
```
http://127.0.0.1:5000
```
Tip: In MLflow, expand the hpo_parent run to see nested Optuna trial runs.

Run the Prediction API (FastAPI)

Start the server:
```
uvicorn app.main:app --reload
```

Open interactive docs:
```
http://127.0.0.1:8000/docs
```
Health check:
```
curl -s http://127.0.0.1:8000/health
```

Schema (required feature keys):
```
curl -s http://127.0.0.1:8000/schema
```

Example prediction (uses a real row from the same built-in dataset):
```
python - <<'PY'
import requests
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer(as_frame=True)
row = ds.data.iloc[0].to_dict()

r = requests.post("http://127.0.0.1:8000/predict", json=row)
print("status:", r.status_code)
print("response:", r.json())
PY
```
Run the Demo UI (Streamlit)

The Streamlit UI is a friendlier demo layer on top of the model:

Click Load example values

Click Predict

View SHAP explanation plots

streamlit run ui/streamlit_app.py

Explainability (SHAP)

These are generated during training in artifacts/shap/.

(Optional) Run Tests
```
pytest -q
```
(Optional) Docker
Build and run API container
docker build -t tabular-ml-service .
docker run -p 8000:8000 tabular-ml-service

Docker Compose (MLflow + API)
docker compose up --build


MLflow: http://127.0.0.1:5000

API: http://127.0.0.1:8000

Notes on What’s Committed

Recommended GitHub hygiene:

Do NOT commit .venv/ or mlruns/

Usually do NOT commit artifacts/model.joblib

It is useful to commit artifacts/shap/*.png so the README renders explainability visuals

