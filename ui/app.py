# ui/app.py

import os
import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Privacy-Preserving ML Dashboard",
    layout="wide",
)

st.title("🔐 Privacy-Preserving ML Dashboard")
st.caption("Client–Server encrypted inference with multi-model support")

# -------------------------------------------------------
# Sidebar: API + Health Check
# -------------------------------------------------------
st.sidebar.header("Backend Status")

if st.sidebar.button("🔄 Check API Health"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            st.sidebar.success(f"Healthy: {resp.json()}")
        else:
            st.sidebar.error(f"Health check failed: {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Error contacting API: {e}")


st.sidebar.markdown("---")
st.sidebar.header("Model Configuration")

model_label_to_type = {
    "SVM": "svm",
    "Logistic Regression": "logreg",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
    "MLP Neural Network": "mlp",
}

model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_label_to_type.keys()),
)

model_type = model_label_to_type[model_name]

kernel = st.sidebar.selectbox(
    "SVM Kernel (only used if model = SVM)",
    ["linear", "rbf", "poly", "sigmoid"],
)

C = st.sidebar.slider("C (Regularization Strength for SVM)", 0.1, 5.0, 1.0, 0.1)
max_rows = st.sidebar.slider("Max Rows (for quick demo)", 10, 300, 50, 10)

st.sidebar.markdown("---")
st.sidebar.write("API URL:")
st.sidebar.code(API_URL)

# -------------------------------------------------------
# File Upload Section
# -------------------------------------------------------
st.markdown("### 1️⃣ Upload / Inspect Data")

col_train, col_test = st.columns(2)

with col_train:
    st.subheader("Server Training Data (public)")
    train_file = st.file_uploader(
        "Upload Train CSV (optional)", type="csv", key="train_uploader"
    )
    if train_file is not None:
        df_train = pd.read_csv(train_file)
        df_train.to_csv("data/server_train.csv", index=False)
        st.success("Uploaded and saved as data/server_train.csv")
        st.dataframe(df_train.head())
    else:
        if os.path.exists("data/server_train.csv"):
            df_train = pd.read_csv("data/server_train.csv").head()
            st.write("Using existing data/server_train.csv")
            st.dataframe(df_train)
        else:
            st.warning("No training CSV found yet. Please upload one.")

with col_test:
    st.subheader("Client Test Data (private)")
    test_file = st.file_uploader(
        "Upload Test CSV (optional)", type="csv", key="test_uploader"
    )
    if test_file is not None:
        df_test = pd.read_csv(test_file)
        df_test.to_csv("data/client_test.csv", index=False)
        st.success("Uploaded and saved as data/client_test.csv")
        st.dataframe(df_test.head())
    else:
        if os.path.exists("data/client_test.csv"):
            df_test = pd.read_csv("data/client_test.csv").head()
            st.write("Using existing data/client_test.csv")
            st.dataframe(df_test)
        else:
            st.warning("No test CSV found yet. Please upload one.")

st.markdown("---")
st.markdown("### 2️⃣ Run Secure Training + Encrypted Inference")

if st.button("🚀 Run Pipeline"):
    with st.spinner("Running secure pipeline on backend..."):
        try:
            payload = {
                "model_type": model_type,
                "kernel": kernel,
                "C": C,
                "max_rows": max_rows,
            }
            resp = requests.post(
                f"{API_URL}/run_pipeline", json=payload, timeout=300
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"Error calling backend: {e}")
            result = None

    if result is not None:
        st.success("Pipeline completed successfully!")

        # Basic metrics
        st.subheader("Overall Performance")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Accuracy", f"{result['accuracy']:.4f}")
        with metric_col2:
            st.metric("Samples", result["n_samples"])
        with metric_col3:
            st.metric("Model", result["model_type"])

        st.write(
            f"Model: **{model_name}**, Kernel: `{result.get('kernel', '-')}`, C: `{result.get('C', '-')}`"
        )

        # Predictions table
        st.subheader("Predictions vs True Labels")
        df_results = pd.DataFrame(
            {
                "y_true": result["y_true"],
                "y_pred": result["y_pred"],
            }
        )
        st.dataframe(df_results)

        # ---------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = np.array(result["confusion_matrix"])
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm)

        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_title("Confusion Matrix")

        for (i, j), v in np.ndenumerate(cm):
            ax_cm.text(j, i, str(v), ha="center", va="center")

        st.pyplot(fig_cm)

        # ---------------------------------------------------
        # ROC Curve
        # ---------------------------------------------------
        st.subheader("ROC Curve")

        fpr = np.array(result["roc_fpr"])
        tpr = np.array(result["roc_tpr"])

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label="ROC Curve")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")

        st.pyplot(fig_roc)

        # ---------------------------------------------------
        # Feature Importance
        # ---------------------------------------------------
        st.subheader("Feature Importance")

        fi = result.get("feature_importances", None)
        if fi:
            df_fi = pd.DataFrame(fi)
            df_fi = df_fi.sort_values("importance", ascending=False)
            st.bar_chart(df_fi.set_index("feature"))
            st.dataframe(df_fi)
        else:
            st.info(
                "Feature importance not available for this model (e.g., SVM/MLP without clear feature importances)."
            )

st.markdown("---")
st.markdown("### 3️⃣ Architecture Overview")

st.write(
    """
This dashboard communicates with a FastAPI backend that runs the privacy-preserving ML pipeline:

1. **Client (this UI)** can upload:
   - Public training data (server side)
   - Private test data (client side)

2. **Backend (FastAPI)**:
   - Calls the Paillier-based crypto pipeline
   - Trains selected model: SVM / Logistic Regression / Random Forest / Gradient Boosting / MLP
   - Runs encrypted inference
   - Returns predictions, confusion matrix, ROC curve points, and feature importance.

3. **UI**:
   - Visualizes metrics, confusion matrix, ROC, and feature importances.
"""
)
