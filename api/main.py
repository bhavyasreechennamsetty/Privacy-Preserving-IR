# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse

from core.crypto_pipeline import run_secure_pipeline

app = FastAPI(
    title="Privacy Preserving ML API",
    description="Secure client-server ML with Paillier encryption",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Privacy-Preserving ML API</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    text-align: center; 
                    padding: 40px; 
                    background: #f5f5f5;
                }
                h1 { color: #2c3e50; }
                p { color: #333; font-size: 18px; }
                a.button {
                    display: inline-block;
                    margin: 20px;
                    padding: 12px 24px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-size: 18px;
                }
                a.button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>🔐 Privacy-Preserving ML API</h1>
            <p>Welcome! This backend powers secure, client-server machine learning using Paillier Homomorphic Encryption.</p>
            <a class="button" href="/docs">Open API Docs</a>
            <a class="button" href="/health">Health Check</a>
        </body>
    </html>
    """


class RunConfig(BaseModel):
    model_type: Optional[str] = "svm"   # svm, logreg, rf, gb, mlp
    kernel: Optional[str] = "linear"    # used only for SVM
    C: Optional[float] = 1.0
    max_rows: Optional[int] = 50


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run_pipeline")
def run_pipeline(cfg: RunConfig):
    """
    Trigger the full secure pipeline.
    """
    result = run_secure_pipeline(
        train_path="data/server_train.csv",
        test_path="data/client_test.csv",
        target_col="Outcome",
        model_type=cfg.model_type,
        kernel=cfg.kernel,
        C=cfg.C,
        max_rows=cfg.max_rows,
    )
    return result
