from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import pandas as pd

def make_shap_artifacts(model, X_sample: pd.DataFrame, out_dir: Path) -> dict:
    #Creates SHAP summary plot artifacts for a tree model (LightGBM).
    #Returns paths/metadata.

    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap may return list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Bar summary
    plt.figure()
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
    bar_path = out_dir / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # Beeswarm summary
    plt.figure()
    shap.summary_plot(shap_vals, X_sample, show=False)
    swarm_path = out_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(swarm_path, dpi=200)
    plt.close()

    meta = {
        "n_rows_explained": int(X_sample.shape[0]),
        "artifacts": {
            "shap_summary_bar": str(bar_path),
            "shap_summary_beeswarm": str(swarm_path),
        },
    }
    (out_dir / "shap_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
