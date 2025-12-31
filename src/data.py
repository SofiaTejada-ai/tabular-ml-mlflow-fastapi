from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import load_breast_cancer

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    target_name: str

def load_default() -> Dataset:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y = ds.target.copy()
    return Dataset(
        X=X,
        y=y,
        feature_names=list(X.columns),
        target_name="target",
    )
