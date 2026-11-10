"""
ml/check_window_structure.py

Feasibility spike, step 8. The CICIoT2023 paper (Neto et al. 2023, section 4)
says each CSV row is the MEAN of per-packet feature values over a window of
10 packets (benign traffic and 14 attack types) or 100 packets (all DDoS, DoS
and Mirai types). That gives a testable explanation for what profile_variance.py
showed (Variance takes very few distinct values, is exactly 0 whenever Std is
0, and has median 0.900 on benign rows):

    Hypothesis H*: per-packet Variance is 0 or 1 (1 when the flow's packet
    lengths are not all equal, 0 otherwise), and the row is the window mean.
    If true, Variance needs NO real incoming/outgoing direction at all.

Part A-D (no models, about a minute) test H* and related guesses.
Part E (loads the frozen models) checks a second consequence of the window
sizes: because the window size follows the attack label, the headline AUC may
be driven mostly by the 100-packet flood rows. It reports detection separately
for the 10-packet and 100-packet groups.

Everything printed is evidence, not proof. Nothing is saved.

Run from project root:   python ml/check_window_structure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR, SEED

# Paper section 4: windows of 10 packets for BenignTraffic plus these labels.
# Every other label (all DDoS-*, DoS-*, Mirai-*) uses windows of 100 packets.
WINDOW_10_LABELS = [
    "Backdoor_Malware", "BrowserHijacking", "CommandInjection",
    "DictionaryBruteForce", "DNS_Spoofing", "MITM-ArpSpoofing",
    "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan",
    "SqlInjection", "Uploading_Attack", "VulnerabilityScan", "XSS",
]

STD_EPS = 1e-3  # bigger than before: float32 round-trips leave ~1e-5 noise

cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
ix = {c: i for i, c in enumerate(cols)}
scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
rng = np.random.default_rng(SEED)


def to_raw(X_scaled):
    return scaler.inverse_transform(X_scaled.astype("float64"))


def col(D, name):
    return D[:, ix[name]]


def group_array(labels):
    lab = np.asarray(labels).astype(str)
    return np.where(lab == BENIGN_LABEL, "benign",
                    np.where(np.isin(lab, WINDOW_10_LABELS), "attack-10", "attack-100"))


benign = to_raw(np.load(PROCESSED_DIR / "X_train_benign.npy"))
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
grp_all = group_array(y_val)

# Safety check on the label lists above
w100_labels = np.unique(np.asarray(y_val).astype(str)[grp_all == "attack-100"])
odd = sorted(l for l in w100_labels if not l.startswith(("DDoS", "DoS", "Mirai")))
if odd:
    print("WARNING: labels treated as window-100 that are not DDoS/DoS/Mirai:", odd)

pick = rng.choice(len(X_val), size=300_000, replace=False)
mixed = to_raw(X_val[pick])
grp = grp_all[pick]

# ---------------------------------------------------------------- Part A
print("=== A. Is Variance a window mean of 0/1 values? ===")
print("If each per-packet Variance is 0 or 1, a mean over W packets is a multiple of 1/W.")


def share_multiple(v, k, tol=1e-3):
    return float(np.mean(np.abs(k * v - np.round(k * v)) <= tol))


for title, v, k in [
    ("benign train (window 10)", col(benign, "Variance"), 10),
    ("benign validation sample", col(mixed, "Variance")[grp == "benign"], 10),
    ("attack-10 validation sample", col(mixed, "Variance")[grp == "attack-10"], 10),
    ("attack-100 validation sample", col(mixed, "Variance")[grp == "attack-100"], 100),
]:
    print(f"  {title:<30}{len(v):>9,} rows: share that are multiples of 1/{k}: "
          f"{share_multiple(v, k):6.1%}")

vals, counts = np.unique(np.round(col(benign, "Variance"), 3), return_counts=True)
top = np.argsort(-counts)[:8]
print("  most common benign-train Variance values (3 dp): "
      + ", ".join(f"{vals[i]:.3f} ({counts[i] / counts.sum():.1%})" for i in top))

# ---------------------------------------------------------------- Part B
print("\n=== B. Variance vs Std (Std > 0 means the flow's lengths are not all equal) ===")
for title, D in [("benign train", benign), ("mixed validation", mixed)]:
    s, v = col(D, "Std"), col(D, "Variance")
    zero = s <= STD_EPS
    print(f"  {title:<18} Std~0: {zero.sum():>8,} rows, Variance~0 on {np.mean(v[zero] <= 1e-3):6.1%}"
          f"  |  Std>0: {(~zero).sum():>8,} rows, Variance>0 on {np.mean(v[~zero] > 1e-3):6.1%}")

# ---------------------------------------------------------------- Part C
print("\n=== C. Covariance vs Std^2 (identical in/out sets would make them equal) ===")
for title, D in [("benign train", benign), ("mixed validation", mixed)]:
    s, c = col(D, "Std"), col(D, "Covariance")
    ok = s > STD_EPS
    r = c[ok] / s[ok] ** 2
    med = float(np.median(r))
    print(f"  {title:<18} rows={ok.sum():>8,}  median ratio={med:>8.4g}  "
          f"p5..p95=[{np.percentile(r, 5):.3g}, {np.percentile(r, 95):.3g}]  "
          f"within 10% of median={np.mean(np.abs(r / med - 1) <= 0.1):6.1%}  "
          f"spearman={spearmanr(c[ok], s[ok] ** 2)[0]:.3f}")

# ---------------------------------------------------------------- Part D
print("\n=== D. Same features in the three groups (window size follows the label) ===")
print("  median [p5, p95] on the mixed validation sample")
for f in ["Number", "Std", "AVG", "Tot sum", "Variance"]:
    cells = []
    for g_name in ["benign", "attack-10", "attack-100"]:
        x = col(mixed, f)[grp == g_name]
        cells.append(f"{g_name}: {np.median(x):.3g} [{np.percentile(x, 5):.3g}, "
                     f"{np.percentile(x, 95):.3g}]")
    print(f"  {f:<9}" + "   ".join(cells))

# ---------------------------------------------------------------- Part E
print("\n=== E. Detection by window group (frozen saved models, ROC-AUC) ===")
from tensorflow import keras  # noqa: E402  (imported late so Parts A-D stay fast)

from ensemble.risk_engine import LOW_MEDIUM_THRESHOLD, fuse  # noqa: E402
from ensemble.score_normalization import ScoreNormalizer  # noqa: E402

rng2 = np.random.default_rng(SEED)
idx_b = np.where(grp_all == "benign")[0]
idx_10 = np.where(grp_all == "attack-10")[0]
idx_100 = rng2.choice(np.where(grp_all == "attack-100")[0], size=100_000, replace=False)
idx = np.concatenate([idx_b, idx_10, idx_100])
g = np.concatenate([np.zeros(len(idx_b)), np.ones(len(idx_10)),
                    np.full(len(idx_100), 2)]).astype(int)
X = X_val[idx]
print(f"  rows: benign {len(idx_b):,}, attack-10 {len(idx_10):,}, "
      f"attack-100 (sampled) {len(idx_100):,}")

if_model = joblib.load(SAVED_MODELS_DIR / "isolation_forest.joblib")
ae_model = keras.models.load_model(SAVED_MODELS_DIR / "autoencoder.keras")
gmm_model = joblib.load(SAVED_MODELS_DIR / "gmm.joblib")
normalizer = ScoreNormalizer.load(SAVED_MODELS_DIR / "norm_bounds.json")

recon = ae_model.predict(X, verbose=0)
raw = {
    "isolation_forest": if_model.decision_function(X),
    "autoencoder": np.mean(np.square(X - recon), axis=1),
    "gmm": gmm_model.score_samples(X),
}
norm = normalizer.transform(raw)
scores = dict(norm)
scores["all3"] = fuse(norm)
scores["AE+GMM"] = fuse({m: norm[m] for m in ("autoencoder", "gmm")})


def auc_vs_benign(score, attack_group):
    m = (g == 0) | (g == attack_group)
    return roc_auc_score((g[m] == attack_group).astype(int), score[m])


print(f"\n  {'score':<18}{'AUC benign vs attack-10':>26}{'AUC benign vs attack-100':>27}")
for name, s in scores.items():
    print(f"  {name:<18}{auc_vs_benign(s, 1):>26.3f}{auc_vs_benign(s, 2):>27.3f}")

flag = scores["all3"] >= LOW_MEDIUM_THRESHOLD
print(f"\n  At the production threshold ({LOW_MEDIUM_THRESHOLD}), all-three fusion flags: "
      f"benign {flag[g == 0].mean():.1%} (false alarms), "
      f"attack-10 {flag[g == 1].mean():.1%}, attack-100 {flag[g == 2].mean():.1%}")
print("\nIf AUC on attack-10 is far below attack-100, the headline AUC is driven mostly "
      "by the 100-packet flood rows, and the report must say so.")