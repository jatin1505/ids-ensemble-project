"""Load frozen model artifacts once and score one raw CICIoT2023 row."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from ensemble.risk_engine import build_risk_event
from ensemble.score_normalization import ScoreNormalizer
from shared.schemas import RiskEvent

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "ml" / "saved_models"

_runtime: dict[str, object] | None = None


def load_runtime() -> None:
    """Load all immutable artifacts required for live inference exactly once."""
    global _runtime
    if _runtime is not None:
        return

    required = (
        "scaler.joblib", "feature_columns.json", "isolation_forest.joblib",
        "autoencoder.keras", "gmm.joblib", "norm_bounds.json",
    )
    missing = [name for name in required if not (MODELS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot start replay: missing trained artifacts in "
            f"{MODELS_DIR}: {', '.join(missing)}"
        )

    _runtime = {
        "scaler": joblib.load(MODELS_DIR / "scaler.joblib"),
        "feature_columns": json.loads((MODELS_DIR / "feature_columns.json").read_text()),
        "isolation_forest": joblib.load(MODELS_DIR / "isolation_forest.joblib"),
        "autoencoder": keras.models.load_model(MODELS_DIR / "autoencoder.keras"),
        "gmm": joblib.load(MODELS_DIR / "gmm.joblib"),
        "normalizer": ScoreNormalizer.load(MODELS_DIR / "norm_bounds.json"),
    }


def score_flow(
    row: pd.Series,
    *,
    flow_id: str,
    timestamp: float,
    protocol: str | None = None,
    attack_category: str | None = None,
) -> RiskEvent:
    """Return a normalized and fused risk event for one raw CSV row."""
    if _runtime is None:
        raise RuntimeError("Runtime is not loaded; call load_runtime() first.")

    feature_columns: list[str] = _runtime["feature_columns"]  # type: ignore[assignment]
    missing_features = [column for column in feature_columns if column not in row.index]
    if missing_features:
        raise ValueError(f"Flow is missing required features: {missing_features[:5]}")

    raw_features = row.loc[feature_columns].to_numpy(dtype="float64").reshape(1, -1)
    scaler = _runtime["scaler"]
    features = scaler.transform(raw_features).astype("float32")  # type: ignore[union-attr]
    if_model = _runtime["isolation_forest"]
    autoencoder = _runtime["autoencoder"]
    gmm = _runtime["gmm"]
    normalizer: ScoreNormalizer = _runtime["normalizer"]  # type: ignore[assignment]

    reconstruction = autoencoder.predict(features, verbose=0)  # type: ignore[union-attr]
    raw_scores = {
        "isolation_forest": float(if_model.decision_function(features)[0]),  # type: ignore[union-attr]
        "autoencoder": float(np.mean(np.square(features - reconstruction))),
        "gmm": float(gmm.score_samples(features)[0]),  # type: ignore[union-attr]
    }
    return build_risk_event(
        flow_id=flow_id,
        timestamp=timestamp,
        normalized_scores=normalizer.transform(raw_scores),
        protocol=protocol,
        attack_category=attack_category,
    )
