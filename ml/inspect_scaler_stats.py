"""
ml/inspect_scaler_stats.py

Feasibility spike, step 2. Prints, for each of the 46 features, the range of
RAW (unscaled) values seen in benign training data. Any live-extracted feature
that lands far outside these ranges will produce an extreme z-score after
scaling, and the models will call it anomalous no matter what the traffic is.

Run from ml/:   python inspect_scaler_stats.py
"""
import json

import joblib
import numpy as np

from config import PROCESSED_DIR, SAVED_MODELS_DIR

scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
X_scaled = np.load(PROCESSED_DIR / "X_train_benign.npy")

# Undo the scaling to get back to the original units (bytes, packets/sec, ...)
raw = scaler.inverse_transform(X_scaled)
p1, p50, p99 = np.percentile(raw, [1, 50, 99], axis=0)

print(f"{'feature':<18}{'p1':>11}{'median':>11}{'p99':>13}{'max':>13}   note")
for i, name in enumerate(cols):
    # StandardScaler stores variance; 0 means the column never changed in benign data
    note = "CONSTANT in benign" if scaler.var_[i] == 0 else ""
    print(f"{name:<18}{p1[i]:>11.3g}{p50[i]:>11.3g}{p99[i]:>13.3g}{raw[:, i].max():>13.3g}   {note}")