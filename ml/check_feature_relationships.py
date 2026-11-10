"""
ml/check_feature_relationships.py

Feasibility spike, step 5. Retraining showed the four "Group D" features
(Magnitue, Radius, Covariance, Variance) carry real signal: dropping them cost
the autoencoder 0.990 -> 0.959 and Isolation Forest 0.916 -> 0.841. A live
extractor therefore has to supply them. This script uses only the dataset
itself (no PCAPs) to answer two questions:

  Part 1. Which algebraic relationships hold between CICIoT2023 columns?
          This tells us what the features actually MEAN (for example, whether
          Radius is a fixed multiple of Std), which tells us how to compute
          them from packets.
  Part 2. Can each Group D feature be PREDICTED from the 22 simple features
          we can already reproduce? This is a fallback if the exact formulas
          stay unclear.

Every relationship below is a hypothesis to TEST, not a fact. The script only
prints evidence. Member A decides what it means, after checking the
CICIoT2023 paper's feature table.

Run from project root:   python ml/check_feature_relationships.py
Nothing is saved; ml/saved_models is untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR, SEED

CONSTANTS = ["Drate", "ece_flag_number", "cwr_flag_number",
             "Telnet", "SMTP", "SSH", "IRC", "DHCP"]
UNREPRODUCIBLE = ["IAT", "Weight", "Header_Length", "urg_count", "rst_count",
                  "ack_count", "syn_count", "fin_count", "flow_duration",
                  "Number", "Rate", "Srate"]
GROUP_D = ["Magnitue", "Radius", "Covariance", "Variance"]

cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
ix = {c: i for i, c in enumerate(cols)}
scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
rng = np.random.default_rng(SEED)

TOL = 1e-6  # raw values like -3.5e-13 are float noise from un-scaling, i.e. 0


def to_raw(X_scaled):
    """Undo StandardScaler so we look at real units (bytes, TTL, ...)."""
    return scaler.inverse_transform(X_scaled.astype("float64"))


def col(D, name):
    return D[:, ix[name]]


# Benign training rows (what the models learned) and a mixed validation
# sample (benign + attack), so we can see if a relationship is structural
# (holds for all traffic) or only a benign habit.
benign = to_raw(np.load(PROCESSED_DIR / "X_train_benign.npy"))
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
pick = rng.choice(len(X_val), size=200_000, replace=False)
mixed = to_raw(X_val[pick])
mixed_is_benign = (y_val[pick] == BENIGN_LABEL)

# ---------------------------------------------------------------- Part 1
# Ratios: if numerator/denominator is nearly constant, a fixed formula links
# the two columns. We report the median ratio and how many rows sit within 5%
# of it, instead of guessing the constant in advance.
RATIOS = [
    ("Magnitue^2 / AVG",
     lambda D: col(D, "Magnitue") ** 2, lambda D: col(D, "AVG")),
    ("Radius / Std",
     lambda D: col(D, "Radius"), lambda D: col(D, "Std")),
    ("Tot sum / (Number * AVG)",
     lambda D: col(D, "Tot sum"), lambda D: col(D, "Number") * col(D, "AVG")),
    ("Weight / Number^2",
     lambda D: col(D, "Weight"), lambda D: col(D, "Number") ** 2),
]


def tol(D):
    return 1e-3 * np.maximum(1.0, np.abs(col(D, "Max")))


# Invariants: statements that must be true if the columns mean what their
# names suggest. The Std bound assumes population std; if it fails for many
# rows, the column may be a sample std or may mix several windows. Either way
# that is informative.
INVARIANTS = [
    ("Min <= AVG <= Max",
     lambda D: (col(D, "Min") - tol(D) <= col(D, "AVG"))
               & (col(D, "AVG") <= col(D, "Max") + tol(D))),
    ("Min <= Tot size <= Max",
     lambda D: (col(D, "Min") - tol(D) <= col(D, "Tot size"))
               & (col(D, "Tot size") <= col(D, "Max") + tol(D))),
    ("Std <= (Max - Min) / 2",
     lambda D: col(D, "Std") <= (col(D, "Max") - col(D, "Min")) / 2 + tol(D)),
    ("0 <= Variance <= 1",
     lambda D: (col(D, "Variance") >= -TOL) & (col(D, "Variance") <= 1 + TOL)),
]


def ratio_check(label, D, num, den):
    n, d = num(D), den(D)
    ok = d > TOL
    r = n[ok] / d[ok]
    med = float(np.median(r))
    close = np.mean(np.abs(r - med) <= 0.05 * abs(med)) if med != 0 else float("nan")
    p5, p95 = np.percentile(r, [5, 95])
    print(f"  {label:<26} rows={ok.sum():>8,}  median={med:>9.4g}  "
          f"p5..p95=[{p5:.3g}, {p95:.3g}]  within 5% of median={close:6.1%}")


print("=== Part 1: relationships between columns ===")
for title, data in [("BENIGN train rows", benign),
                    ("MIXED validation rows (benign + attack)", mixed)]:
    print(f"\n--- {title} ({len(data):,} rows) ---")
    print("Ratios (a near-constant ratio suggests a fixed formula):")
    for label, num, den in RATIOS:
        ratio_check(label, data, num, den)
    print("Invariants (share of rows where the statement is true):")
    for label, fn in INVARIANTS:
        print(f"  {label:<26} {np.mean(fn(data)):6.1%}")

# ---------------------------------------------------------------- Part 2
REPRODUCIBLE = [c for c in cols if c not in CONSTANTS + UNREPRODUCIBLE + GROUP_D]
r_idx = [ix[c] for c in REPRODUCIBLE]
train_rows = rng.choice(len(benign), size=min(60_000, len(benign)), replace=False)

print(f"\n=== Part 2: predict each Group D feature from the "
      f"{len(REPRODUCIBLE)} reproducible ones ===")
print("Trained on benign train rows, tested on held-out validation rows.")
print("spearman: 1.0 = predictions rank rows perfectly.  "
      "medRelErr: median |pred - true| / |true| (rows where |true| > 1e-3).\n")
print(f"{'target':<12}{'test rows':<10}{'spearman':>10}{'medRelErr':>11}")

for target in GROUP_D:
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=SEED)
    reg.fit(benign[train_rows][:, r_idx], benign[train_rows, ix[target]])
    for subset, mask in [("benign", mixed_is_benign), ("attack", ~mixed_is_benign)]:
        X_te = mixed[mask][:, r_idx]
        true = mixed[mask][:, ix[target]]
        pred = reg.predict(X_te)
        rho = spearmanr(pred, true)[0]
        big = np.abs(true) > 1e-3
        rel = (np.median(np.abs(pred[big] - true[big]) / np.abs(true[big]))
               if big.any() else float("nan"))
        print(f"{target:<12}{subset:<10}{rho:>10.3f}{rel:>11.3f}")

print("\nReminder: these are hypotheses and statistics, not proof. Check every "
      "conclusion against the CICIoT2023 paper before writing it into "
      "docs/FEATURE_DEFINITIONS.md.")