import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Iris classifier", layout="centered")

TITLE = "Iris classifier (StandardScaler + LogisticRegression)"
st.title(TITLE)
st.write("Load the `iris_model.pkl` artifact (created in Colab) and try predictions.")

@st.cache_resource
def load_artifact(path: str = "iris_model.pkl"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at: {p.resolve()}")
    artifact = joblib.load(p)
    return artifact

# Sidebar upload
st.sidebar.header("Model (iris_model.pkl)")
uploaded = st.sidebar.file_uploader("Upload iris_model.pkl (optional)", type=["pkl", "joblib"])

artifact = None
if uploaded is not None:
    try:
        artifact = joblib.load(uploaded)
        st.sidebar.success("Artifact loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded file: {e}")

if artifact is None:
    try:
        artifact = load_artifact("iris_model.pkl")
        st.sidebar.success("Loaded artifact from repo root: iris_model.pkl")
    except FileNotFoundError:
        st.sidebar.warning("Upload iris_model.pkl or add it to the repo root.")
    except Exception as e:
        st.sidebar.error(f"Error loading artifact: {e}")

if artifact is None:
    st.info("Upload the model file to continue.")
    st.stop()

pipeline = artifact.get("pipeline")
feature_names = list(artifact.get("feature_names"))
target_names = list(artifact.get("target_names"))

st.write(f"**Features used:** {', '.join(feature_names)}")
st.write(f"**Target classes:** {', '.join(target_names)}")

# ---------------- Prediction UI ----------------
st.header("Make a prediction")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(feature_names[0], min_value=0.0, value=5.1, step=0.1)
    sepal_width  = st.number_input(feature_names[1], min_value=0.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input(feature_names[2], min_value=0.0, value=1.4, step=0.1)
    petal_width  = st.number_input(feature_names[3], min_value=0.0, value=0.2, step=0.1)

sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
sample_df = pd.DataFrame(sample, columns=feature_names)

if st.button("Predict"):
    pred = pipeline.predict(sample_df)[0]
    probs = pipeline.predict_proba(sample_df)[0]

    st.success(f"Predicted class: **{pred}**")

    proba_df = pd.DataFrame({
        "class": target_names,
        "probability": probs
    }).sort_values("probability", ascending=False)
    
    st.subheader("Class probabilities")
    st.dataframe(proba_df)

st.markdown("---")
st.caption("Iris Model Predictor — Streamlit App")
