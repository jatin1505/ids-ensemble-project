"""
ml/export_live_fill_values.py

Task A3 of the live-capture extension (docs/LIVE_CAPTURE_HANDOFF.md).

The live extractor cannot measure 13 of the 46 features (the 12 unreproducible
ones plus Covariance, decisions D1/D2). Instead of inventing values, the
backend fills each one with the BENIGN MEDIAN from the training data, in raw
units. That is exactly what the ablation experiments did, so live scoring
matches the validated "neutralized" scenarios.

This script computes those 13 medians once and writes them to
live/live_fill_values.json. The file is 13 numbers, so it is committed to git
(live/ is not git-ignored, unlike data/ and ml/saved_models/).

Medians are unchanged by StandardScaler's linear transform (the scale is
positive), so computing them in raw units here is equivalent to taking the
median of the scaled matrix, which is what the ablations used.

Run from the project root, after ml/preprocessing.py has produced
data/processed/X_train_benign.npy and ml/saved_models/scaler.joblib:

    python ml/export_live_fill_values.py

Input:  data/processed/X_train_benign.npy
        ml/saved_models/scaler.joblib
        ml/saved_models/feature_columns.json
Output: live/live_fill_values.json   ({feature name: raw benign median})
"""

import json

import joblib
import numpy as np

from config import PROCESSED_DIR, ROOT, SAVED_MODELS_DIR

# The 12 unreproducible features plus Covariance. If the A1 source-code check
# changes which features can be reproduced, this list (and decision D2) is the
# one place to update.
NEUTRALIZED = [
    "IAT", "Weight", "Header_Length", "urg_count", "rst_count",
    "ack_count", "syn_count", "fin_count", "flow_duration",
    "Number", "Rate", "Srate",
    "Covariance",
]

OUTPUT_PATH = ROOT / "live" / "live_fill_values.json"


def main():
    feature_cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())

    missing = [name for name in NEUTRALIZED if name not in feature_cols]
    if missing:
        raise SystemExit(f"Not found in feature_columns.json: {missing}")

    scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
    X_scaled = np.load(PROCESSED_DIR / "X_train_benign.npy")
    print(f"Loaded benign training matrix: {X_scaled.shape}")

    # Back to raw units (bytes, packets/sec, ...), where the live extractor
    # and the fill values both live.
    raw = scaler.inverse_transform(X_scaled.astype("float64"))
    medians = np.median(raw, axis=0)

    fill = {}
    for name in NEUTRALIZED:
        value = float(medians[feature_cols.index(name)])
        # Un-scaling leaves float noise like -3.5e-13 where the true value is 0.
        fill[name] = 0.0 if abs(value) < 1e-9 else value

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(fill, indent=2) + "\n")

    print(f"\nSaved {len(fill)} fill values to {OUTPUT_PATH}")
    for name, value in fill.items():
        print(f"  {name:<14}{value:>16.6g}")


if __name__ == "__main__":
    main()