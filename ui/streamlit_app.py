import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_names.json"
SHAP_DIR = ARTIFACT_DIR / "shap"
SHAP_BAR = SHAP_DIR / "shap_summary_bar.png"
SHAP_SWARM = SHAP_DIR / "shap_summary_beeswarm.png"
SHAP_META = SHAP_DIR / "shap_meta.json"

st.set_page_config(page_title="Tabular Predictor", page_icon="", layout="wide")
st.title("Prediction Demo (End-to-End ML System)")

# -----------------------------
# Plain-English explanation UI
# -----------------------------
with st.sidebar:
    st.header("What is this?")
    st.write(
        "This is a demo app for a machine-learning model. "
        "The model takes a row of numbers (features) and returns a prediction and confidence."
    )
    st.write(
        "**Why the inputs look weird:** In a real product, a user would NOT type these. "
        "These would come automatically from a form, database, or uploaded file."
    )
    st.write(
        "**What to do:** Click **Load example values** (instant demo), then click **Predict**."
    )

# -----------------------------
# Load artifacts
# -----------------------------
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Missing model artifacts. Run training first to create artifacts/model.joblib and artifacts/feature_names.json.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = json.loads(FEATURES_PATH.read_text())

# -----------------------------
# Example row helper
# -----------------------------
def try_get_example_row(feature_names: list[str]) -> dict:
    """
    Returns a realistic example row for the default sklearn breast cancer dataset,
    if available. If not, returns zeros for all features.
    """
    try:
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer(as_frame=True)
        row = ds.data.iloc[0].to_dict()
        # Ensure keys match expected features
        out = {}
        for f in feature_names:
            out[f] = float(row.get(f, 0.0))
        return out
    except Exception:
        return {f: 0.0 for f in feature_names}

# Initialize session state values once
if "values" not in st.session_state:
    st.session_state["values"] = {f: 0.0 for f in features}

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("Try a prediction")

    tab1, tab2 = st.tabs(["Manual input (demo)", "Upload CSV (realistic)"])

    with tab1:
        st.write(
            "This is mainly for demo/testing. If you don’t know what these mean, "
            "click **Load example values** and then **Predict**."
        )

        if st.button("Load example values", use_container_width=True):
            st.session_state["values"] = try_get_example_row(features)
            st.success("Loaded example values. Scroll down and click Predict.")

        # Build inputs (pre-filled from session_state)
        new_values = {}
        with st.expander("Show input fields", expanded=True):
            for f in features:
                default = float(st.session_state["values"].get(f, 0.0))
                new_values[f] = st.number_input(f, value=default)

        st.session_state["values"] = new_values

        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

        if st.button("Predict (manual)", type="primary", use_container_width=True):
            X = pd.DataFrame([st.session_state["values"]], columns=features)
            proba = float(model.predict_proba(X)[:, 1][0])
            pred = int(proba >= threshold)

            st.success(f"Prediction: {pred}")
            st.info(
                f"Probability positive: {proba:.6f}  |  "
                f"{proba*100:.4f}%  |  Threshold: {threshold:.2f}"
            )

    with tab2:
        st.write("Upload a CSV containing **at least** the required feature columns. The app uses the first row.")
        file = st.file_uploader("Upload CSV", type=["csv"])
        threshold = st.slider("Decision threshold (CSV)", 0.05, 0.95, 0.50, 0.01)

        if file is not None:
            df = pd.read_csv(file)
            missing = [f for f in features if f not in df.columns]
            if missing:
                st.error(f"CSV missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            else:
                st.write("Preview (first row):")
                st.dataframe(df[features].head(1), use_container_width=True)

                if st.button("Predict (CSV)", type="primary", use_container_width=True):
                    X = df[features].head(1)
                    proba = float(model.predict_proba(X)[:, 1][0])
                    pred = int(proba >= threshold)

                    st.success(f"Prediction: {pred}")
                    st.info(
                        f"Probability positive: {proba:.6f}  |  "
                        f"{proba*100:.4f}%  |  Threshold: {threshold:.2f}"
                    )

with right:
    st.subheader("Why did the model predict that? (Explainability)")

    if SHAP_BAR.exists() and SHAP_SWARM.exists():
        st.write(
            "These plots are created during training using SHAP. "
            "They show which input features mattered most overall."
        )
        st.image(str(SHAP_BAR), caption="SHAP summary (feature importance)", use_container_width=True)
        st.image(str(SHAP_SWARM), caption="SHAP beeswarm (direction + impact)", use_container_width=True)

        if SHAP_META.exists():
            try:
                meta = json.loads(SHAP_META.read_text())
                st.caption(f"Explained using a sample of {meta.get('n_rows_explained', 'N/A')} rows.")
            except Exception:
                pass
    else:
        st.warning(
            "SHAP images not found yet. Run training first so it generates artifacts/shap/*.png."
        )

st.divider()
st.caption(
    "Tip: For a real-user product, you would replace these raw feature inputs with a simpler form "
    "(plain-English questions) or pull the values from an upstream system automatically."
)
