from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from config import PROCESSED_X_PATH, PROCESSED_Y_PATH, RANDOM_STATE


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    logreg = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
        )),
    ])
    models["logreg"] = logreg

    rf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    models["rf"] = rf

    return models

def load_processed_data():
    X = np.load(PROCESSED_X_PATH)
    y = np.load(PROCESSED_Y_PATH)
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    print(f"\n========== Model: {name} ==========")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    metrics = {}

    metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
    metrics["train_f1"] = f1_score(y_train, y_train_pred, average="binary", pos_label=1)

    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        try:
            metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)
        except ValueError:
            metrics["train_roc_auc"] = float("nan")

    print("--- Training metrics ---")
    print(f"  accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  f1:       {metrics['train_f1']:.4f}")
    print(f"  roc_auc:  {metrics.get('train_roc_auc', float('nan')):.4f}")

    y_pred = model.predict(X_test)
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["f1"] = f1_score(y_test, y_pred, average="binary", pos_label=1)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    print("\n--- Test metrics ---")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  f1:       {metrics['f1']:.4f}")
    print(f"  roc_auc:  {metrics.get('roc_auc', float('nan')):.4f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred))

    return metrics


def run_training():
    print("Loading processed dataset...")
    X, y = load_processed_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    models = build_models(random_state=RANDOM_STATE)

    all_results: Dict[str, dict] = {}
    for name, model in models.items():
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        all_results[name] = metrics

    print("\n=========== Summary ===========")
    for name, metrics in all_results.items():
        acc = metrics.get("accuracy", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        auc = metrics.get("roc_auc", float("nan"))
        print(f"{name}: acc={acc:.4f}, f1={f1:.4f}, roc_auc={auc:.4f}")

    return all_results