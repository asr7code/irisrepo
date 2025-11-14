import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Iris classifier", layout="centered")

st.title("🌸 Iris Classifier (Logistic Regression)")
st.write("This app uses the trained Iris model (`iris_model.pkl`) stored in this repository.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_artifact():
    model_path = Path("iris_model.pkl")
    if not model_path.exists():
        st.error("❌ Could not find `iris_model.pkl` in the repository root.")
        st.stop()
    return joblib.load(model_path)

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

st.success("✅ Model loaded successfully from repository.")


# ---------------- Prediction UI ----------------
st.header("🔮 Make a Prediction")

cols = st.columns(2)

with cols[0]:
    sepal_length = st.number_input(feature_names[0], min_value=0.0, value=5.1, step=0.1)
    sepal_width  = st.number_input(feature_names[1], min_value=0.0, value=3.5, step=0.1)

with cols[1]:
    petal_length = st.number_input(feature_names[2], min_value=0.0, value=1.4, step=0.1)
    petal_width  = st.number_input(feature_names[3], min_value=0.0, value=0.2, step=0.1)

sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=feature_names)

if st.button("Predict"):
    pred = pipeline.predict(sample)[0]
    probs = pipeline.predict_proba(sample)[0]

    st.subheader("🌼 Prediction Result")
    st.success(f"Predicted Species: **{pred}**")

    proba_df = pd.DataFrame({
        "Class": target_names,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    st.subheader("📊 Class Probabilities")
    st.dataframe(proba_df)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Iris Model Deployment — Streamlit App using model stored in the GitHub repo.")
