from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def _select_features(df: pd.DataFrame, target: str) -> List[str]:
    features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise ValueError("No numeric feature columns found for modeling.")
    return features


def train_ml(csv_path: str, target: str, model_path: str) -> None:
    """Train a RandomForest model to predict the target column from numeric features."""
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")

    features = _select_features(df, target)
    X = df[features].values
    y = df[target].values.astype(float)

    # Split for a quick validation metric
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)

    # Persist model with metadata
    artifact = {
        "model": model,
        "features": features,
        "target": target,
        "metrics": {"r2_val": float(score)},
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def predict_ml(csv_path: str, model_path: str, output_csv_path: str) -> None:
    """Load a trained model and produce predictions for the rows in the CSV."""
    df = pd.read_csv(csv_path)
    artifact = joblib.load(model_path)
    model: RandomForestRegressor = artifact["model"]
    features: List[str] = artifact["features"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required feature columns: {missing}")

    preds = model.predict(df[features].values)
    out_df = df.copy()
    out_df["prediction"] = preds
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)

