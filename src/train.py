from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data import load_default
from src.explain import make_shap_artifacts


ARTIFACT_DIR = Path("artifacts")


def build_pipeline(params: dict) -> Pipeline:
    # Simple pipeline: impute then LGBM
    clf = LGBMClassifier(**params)
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clf),
        ]
    )
    return pipe


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }

    pipe = build_pipeline(params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in cv.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))

    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="tabular-lgbm-optuna")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--mlflow_uri", default="file:./mlruns")
    args = ap.parse_args()

    ds = load_default()
    X, y = ds.X, ds.y

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.log_param("dataset", "sklearn_breast_cancer")
        mlflow.log_param("n_trials", args.trials)

        def _obj(trial):
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                score = objective(trial, X, y)
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_mean_roc_auc", score)
                return score

        study = optuna.create_study(direction="maximize")
        study.optimize(_obj, n_trials=args.trials)

        best_params = study.best_trial.params
        # Add fixed params back in
        best_params.update({"random_state": 42, "n_jobs": -1})

        (ARTIFACT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
        mlflow.log_artifact(str(ARTIFACT_DIR / "best_params.json"))

        # Train best model on full data
        best_pipe = build_pipeline(best_params)
        best_pipe.fit(X, y)

        # Evaluate quickly on training set (for demo); you can add a real holdout if you want
        proba = best_pipe.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        mlflow.log_metric("train_roc_auc", roc_auc_score(y, proba))
        mlflow.log_metric("train_accuracy", accuracy_score(y, pred))

        # Save pipeline for API
        model_path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(best_pipe, model_path)
        mlflow.log_artifact(str(model_path))

        # Save feature names
        feat_path = ARTIFACT_DIR / "feature_names.json"
        feat_path.write_text(json.dumps(ds.feature_names, indent=2))
        mlflow.log_artifact(str(feat_path))

        # SHAP (explain the LGBM model inside the pipeline)
        # We need imputed values for SHAP:
        X_imp = pd.DataFrame(
            best_pipe.named_steps["imputer"].transform(X),
            columns=ds.feature_names,
        )
        X_sample = X_imp.sample(n=min(200, len(X_imp)), random_state=42)

        shap_dir = ARTIFACT_DIR / "shap"
        meta = make_shap_artifacts(best_pipe.named_steps["model"], X_sample, shap_dir)

        # Log SHAP artifacts
        mlflow.log_artifact(str(shap_dir / "shap_summary_bar.png"))
        mlflow.log_artifact(str(shap_dir / "shap_summary_beeswarm.png"))
        mlflow.log_artifact(str(shap_dir / "shap_meta.json"))

        print("Done.")
        print(f"Best CV ROC-AUC: {study.best_value:.4f}")
        print(f"Saved: {model_path}")

if __name__ == "__main__":
    main()
