# core/crypto_pipeline.py

import math
from typing import List, Dict, Any, Tuple

import pandas as pd
import phe.encoding
from phe import paillier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


class ExampleEncodedNumber(phe.encoding.EncodedNumber):
    BASE = 64
    LOG2_BASE = math.log(BASE, 2)


def encrypt_vector(vec, public_key):
    encoded_vector = [ExampleEncodedNumber.encode(public_key, float(v)) for v in vec]
    encrypted_vector = [public_key.encrypt(ev) for ev in encoded_vector]
    return encrypted_vector


def decrypt_vector(vec, private_key):
    decoded_vector = [private_key.decrypt_encoded(x, ExampleEncodedNumber) for x in vec]
    decrypted_vector = [d.decode() for d in decoded_vector]
    return decrypted_vector


class Client:
    def __init__(self, key_length: int = 1024):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_length
        )

    def load_data(
        self,
        train_file: str,
        test_file: str,
        target_feature: str,
        max_rows: int | None = None,
    ):
        """Load train & test data. Optionally limit rows for quick demo."""
        train_data = pd.read_csv(train_file).dropna()
        test_data = pd.read_csv(test_file).dropna()

        if max_rows is not None:
            train_data = train_data.iloc[:max_rows]
            test_data = test_data.iloc[:max_rows]

        feature_cols = [c for c in train_data.columns if c != target_feature]

        X_train = train_data[feature_cols].values.tolist()
        y_train = train_data[target_feature].values.tolist()

        X_test = test_data[feature_cols].values.tolist()
        y_test = test_data[target_feature].values.tolist()

        return X_train, y_train, X_test, y_test, feature_cols

    def encrypt_data(self, X_train, y_train, X_test, y_test):
        X_train_encrypted = [encrypt_vector(x, self.public_key) for x in X_train]
        y_train_encrypted = encrypt_vector(y_train, self.public_key)

        X_test_encrypted = [encrypt_vector(x, self.public_key) for x in X_test]
        y_test_encrypted = encrypt_vector(y_test, self.public_key)

        return X_train_encrypted, y_train_encrypted, X_test_encrypted, y_test_encrypted

    def eval(self, encrypted_predictions, encrypted_ytest):
        y_pred = [
            self.private_key.decrypt_encoded(y, ExampleEncodedNumber).decode()
            for y in encrypted_predictions
        ]

        y_test = [
            self.private_key.decrypt_encoded(y, ExampleEncodedNumber).decode()
            for y in encrypted_ytest
        ]

        y_pred = [round(float(x)) for x in y_pred]
        y_test = [round(float(x)) for x in y_test]

        acc = accuracy_score(y_test, y_pred)
        return acc, y_pred, y_test


class Server:
    def __init__(
        self,
        public_key,
        private_key,
        X_train_encrypted,
        y_train_encrypted,
        X_test_encrypted,
        y_test_encrypted,
    ):
        self.public_key = public_key
        self.private_key = private_key
        self.X_train_encrypted = X_train_encrypted
        self.y_train_encrypted = y_train_encrypted
        self.X_test_encrypted = X_test_encrypted
        self.y_test_encrypted = y_test_encrypted
        self.model = None

    def _decrypt_train(self):
        X_train = [decrypt_vector(x, self.private_key) for x in self.X_train_encrypted]
        y_train = decrypt_vector(self.y_train_encrypted, self.private_key)

        X_train = [[float(v) for v in row] for row in X_train]
        y_train = [float(v) for v in y_train]
        return X_train, y_train

    def _decrypt_test(self):
        X_test = [decrypt_vector(x, self.private_key) for x in self.X_test_encrypted]
        X_test = [[float(v) for v in row] for row in X_test]
        return X_test

    def train_model(
        self,
        model_type: str = "svm",
        kernel: str = "linear",
        C: float = 1.0,
    ):
        X_train, y_train = self._decrypt_train()

        if model_type == "svm":
            clf = SVC(kernel=kernel, C=C, probability=True)
        elif model_type == "logreg":
            clf = LogisticRegression(max_iter=1000)
        elif model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            )
        elif model_type == "gb":
            clf = GradientBoostingClassifier(random_state=42)
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        clf.fit(X_train, y_train)
        self.model = clf
        return clf

    def predict(self):
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        X_test = self._decrypt_test()

        preds = self.model.predict(X_test)

        # Scores for ROC (probabilities if available)
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X_test)[:, 1].tolist()
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_test).tolist()
        else:
            scores = [float(p) for p in preds]

        encrypted_preds = [
            self.public_key.encrypt(
                ExampleEncodedNumber.encode(self.public_key, float(p))
            )
            for p in preds
        ]

        return encrypted_preds, scores


def _compute_feature_importance(model, feature_names: List[str]) -> List[Dict[str, float]] | None:
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Use absolute value of coefficients for importance
        coef = getattr(model, "coef_")
        if coef is not None and len(coef.shape) == 2:
            importances = abs(coef[0])

    if importances is None:
        return None

    importances = list(importances)
    if len(importances) != len(feature_names):
        return None

    return [
        {"feature": f, "importance": float(i)}
        for f, i in zip(feature_names, importances)
    ]


def run_secure_pipeline(
    train_path: str,
    test_path: str,
    target_col: str = "Outcome",
    model_type: str = "svm",
    kernel: str = "linear",
    C: float = 1.0,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """
    High-level function:
    - Client loads data & encrypts
    - Server trains selected model & predicts
    - Client evaluates
    - Compute confusion matrix, ROC, feature importances
    Returns a dict for JSON response / UI.
    """
    client = Client(1024)

    X_train, y_train, X_test, y_test, feature_names = client.load_data(
        train_path, test_path, target_col, max_rows=max_rows
    )

    X_train_enc, y_train_enc, X_test_enc, y_test_enc = client.encrypt_data(
        X_train, y_train, X_test, y_test
    )

    server = Server(
        client.public_key,
        client.private_key,
        X_train_enc,
        y_train_enc,
        X_test_enc,
        y_test_enc,
    )

    server.train_model(model_type=model_type, kernel=kernel, C=C)
    encrypted_preds, scores = server.predict()

    acc, y_pred_plain, y_test_plain = client.eval(encrypted_preds, y_test_enc)

    # Confusion matrix
    cm = confusion_matrix(y_test_plain, y_pred_plain).tolist()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test_plain, scores)
    fpr = fpr.tolist()
    tpr = tpr.tolist()

    # Feature importance
    fi = _compute_feature_importance(server.model, feature_names)

    return {
        "accuracy": acc,
        "y_pred": y_pred_plain,
        "y_true": y_test_plain,
        "n_samples": len(y_test_plain),
        "model_type": model_type,
        "kernel": kernel,
        "C": C,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "confusion_matrix": cm,
        "feature_names": feature_names,
        "feature_importances": fi,
    }
