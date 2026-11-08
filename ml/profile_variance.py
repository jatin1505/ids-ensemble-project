"""
ml/profile_variance.py

Feasibility spike, step 7 (profiling only: no models, nothing saved).

Retraining showed that `Variance` is the one Group D feature the live
extractor MUST compute: dropping it costs the autoencoder 0.993 -> 0.963 and
the GMM 0.991 -> 0.922, and it cannot be predicted from other columns. We know
it always lies in [0, 1] and has median about 0.9, but not how it is built.
This script profiles it so Member A has concrete patterns to check against the
CICIoT2023 paper's definition:

  A. How Variance values are distributed (exactly 0, exactly 1, in between),
     for benign rows and for attack rows with and without Std == 0.
  B. Per attack category: how often Std == 0, and Variance / Covariance there.
  C. Spearman correlation of Variance with other columns.

These are observations, not conclusions. Member A decides what they mean.

Run from project root:   python ml/profile_variance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
from scipy.stats import spearmanr

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR, SEED

cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
ix = {c: i for i, c in enumerate(cols)}
scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
rng = np.random.default_rng(SEED)

EPS = 1e-6  # un-scaled values like -2e-8 are float noise, i.e. 0


def to_raw(X_scaled):
    return scaler.inverse_transform(X_scaled.astype("float64"))


def col(D, name):
    return D[:, ix[name]]


benign = to_raw(np.load(PROCESSED_DIR / "X_train_benign.npy"))
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
pick = rng.choice(len(X_val), size=300_000, replace=False)
mixed = to_raw(X_val[pick])
labels = y_val[pick]

is_benign = labels == BENIGN_LABEL
std = col(mixed, "Std")
var = col(mixed, "Variance")
cov = col(mixed, "Covariance")

BINS = [
    ("== 0", lambda v: v <= EPS),
    ("(0, .5)", lambda v: (v > EPS) & (v < 0.5)),
    ("[.5, .9)", lambda v: (v >= 0.5) & (v < 0.9)),
    ("[.9, 1)", lambda v: (v >= 0.9) & (v < 1 - EPS)),
    ("== 1", lambda v: v >= 1 - EPS),
]


def bin_row(title, v):
    shares = "  ".join(f"{name} {fn(v).mean():6.1%}" for name, fn in BINS)
    distinct = len(np.unique(np.round(v, 4)))
    print(f"  {title:<28}{len(v):>9,} rows | {shares} | distinct values (4 dp): {distinct:,}")


print("=== A. Variance value distribution ===")
bin_row("benign (train)", col(benign, "Variance"))
bin_row("benign (validation sample)", var[is_benign])
bin_row("attack, Std == 0", var[~is_benign & (std <= EPS)])
bin_row("attack, Std > 0", var[~is_benign & (std > EPS)])

print("\n=== B. Per label (sampled validation rows) ===")
print(f"  {'label':<30}{'rows':>8}{'Std==0':>9}{'medVar':>9}{'Var==1':>9}"
      f"{'Var==0':>9}{'medCov':>12}")
for lab in sorted(set(labels), key=lambda l: -(labels == l).sum()):
    m = labels == lab
    v = var[m]
    print(f"  {str(lab):<30}{m.sum():>8,}{(std[m] <= EPS).mean():>9.1%}"
          f"{np.median(v):>9.3f}{(v >= 1 - EPS).mean():>9.1%}"
          f"{(v <= EPS).mean():>9.1%}{np.median(cov[m]):>12.3g}")

print("\n=== C. Spearman correlation of Variance with other columns ===")
print(f"  {'column':<12}{'benign':>9}{'attack':>9}   (nan = column is constant in that subset)")
attack = mixed[~is_benign]
for name in ["Std", "AVG", "Min", "Max", "Tot size", "Tot sum", "Covariance",
             "Magnitue", "Radius", "TCP", "UDP", "ICMP", "HTTPS"]:
    rb = spearmanr(col(benign, name), col(benign, "Variance"))[0]
    ra = spearmanr(col(attack, name), col(attack, "Variance"))[0]
    print(f"  {name:<12}{rb:>9.3f}{ra:>9.3f}")

print("\nThese are observations, not conclusions: check them against the "
      "CICIoT2023 paper's definition of Variance before using them.")