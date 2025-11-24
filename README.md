# Privacy-Preserving Machine Learning with Homomorphic Encryption

This project implements a **client–server privacy-preserving ML system** using:

- **Paillier Homomorphic Encryption**
- **FastAPI** backend
- **Streamlit** dashboard
- **Multi-model support**:
  - SVM
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - MLP Neural Network

The system allows a **client with private test data** to obtain model predictions from a **server that owns a public training dataset**, while preserving privacy.

---

## Features

- Paillier-based encrypted inference
- Client–Server separation (training vs. private evaluation)
- FastAPI backend with `/run_pipeline` & `/health`
-  Streamlit dashboard:
  - Upload train/test CSVs
  - Choose ML model
  - Run secure pipeline
  - View:
    - Accuracy
    - Confusion matrix
    - ROC curve
    - Feature importance
- Tested with `pytest`

---

## Architecture

```mermaid
flowchart LR
    subgraph Client["Client (Streamlit UI)"]
        U[Upload private test.csv<br>and public train.csv]
        Cfg[Choose model & params]
        Req[Call /run_pipeline]
        Viz[Visualize metrics<br>(ROC, CM, FI)]
    end

    subgraph Server["Server (FastAPI + Crypto Pipeline)"]
        API[FastAPI /run_pipeline]
        Core[crypto_pipeline.py<br>(Paillier + ML models)]
        Model[Trained ML Model]
    end

    U --> Cfg --> Req --> API --> Core --> Model
    Model --> Core --> API --> Viz
