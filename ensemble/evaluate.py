"""
ensemble/evaluate.py

Compares each individual model (Isolation Forest, Autoencoder, GMM)
against the fused ensemble on the SAME held-out validation set, using
the SAME normalized scores and the SAME decision rule -- so "does the
ensemble actually help" is an apples-to-apples comparison, not four
different methodologies mashed together.

This is where Objective 3 (ensemble vs. individual model comparison)
gets its numbers for the report. Runs at development time only -- it
has no role in the live WebSocket pipeline.

THIS FILE WAS EMPTY in the repo. First draft to unblock the evaluation
milestone -- AGL should review the threshold choice (see note 2 below)
before these numbers go in the report.

Inputs (must already exist -- see the FileNotFoundError messages below
for how to produce them):
    ml/saved_models/validation_raw_scores.npz   -- from
        ml/export_validation_scores.py: raw scores for all three
        models + y_val (ground-truth attack_category strings,
        including "BenignTraffic")
    ml/saved_models/norm_bounds.json             -- from
        ml/fit_score_normalizer.py: the frozen [0,1] normalization
        bounds fit on this same validation set

Run from the project root:
    python ensemble/evaluate.py

Methodology notes
------------------
1. ROC-AUC, not PR-AUC, is the threshold-free headline number.
   FIX (was wrong in the first version of this file): PR-AUC / average
   precision was computed with attack as the "positive" class. On this
   validation split attack is ~97.7% of rows, i.e. the MAJORITY, not
   the rare class PR-AUC is designed to reward ranking well. A model
   with ZERO actual discriminative power -- pure random scores -- gets
   PR-AUC ~0.977 in this exact configuration purely from that class
   ratio, which is within rounding distance of what real models
   reported here (0.995-0.9997). It was not a meaningful number. ROC-AUC
   is prevalence-invariant (a random model always scores ~0.5,
   regardless of which class is the majority), so it's what's reported
   below.

2. Threshold-based metrics (Precision / Recall / F1 / F2) reuse
   risk_engine.LOW_MEDIUM_THRESHOLD as a single binary cutoff
   ("flagged" = Medium or High, i.e. not Low) for EVERY model,
   individual or fused. That's deliberate: it's the exact decision
   rule the live system actually uses for the ensemble, so individual
   models are being judged by the same yardstick, not a separately
   tuned per-model threshold that would flatter or handicap one of
   them. LOW_MEDIUM_THRESHOLD is still a placeholder (see
   ensemble/risk_engine.py's TODOs) -- rerun this file after AGL
   calibrates the real thresholds.

3. F2 (recall-weighted) is reported alongside F1 per docs/DECISIONS.md:
   a missed attack (false negative) is treated as more costly than a
   false alarm (false positive) for an IDS.

4. The per-attack-category breakdown reports RECALL only (fraction of
   each attack type the ensemble actually flagged). Precision isn't
   meaningful computed within a single-category subset, since every
   row in that subset is already a real attack by construction --
   there's no "false positive within DDoS-SYN_Flood rows" to speak of.

5. A threshold-sweep table for the fused ensemble score. High ROC-AUC
   only means the score RANKS attacks above benign reasonably well in
   aggregate -- it says nothing by itself about whether
   LOW_MEDIUM_THRESHOLD is well-placed. This table exists so that any
   future retraining or threshold change can be re-checked the same
   way #12 in docs/DECISIONS.md was originally derived, rather than
   trusting risk_engine.py's constants without re-verifying them.
"""

import sys
from pathlib import Path

# evaluate.py lives in ensemble/, but needs ml/ (a sibling top-level
# package) for BENIGN_LABEL/SAVED_MODELS_DIR. That only resolves if the
# repo ROOT is on sys.path -- same reasoning as
# ml/fit_score_normalizer.py's sys.path line, mirrored here because
# this file sits on the other side of that same cross-package import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import BENIGN_LABEL, SAVED_MODELS_DIR
from ensemble.risk_engine import LOW_MEDIUM_THRESHOLD, fuse
from ensemble.score_normalization import ScoreNormalizer

SCORES_PATH = SAVED_MODELS_DIR / "validation_raw_scores.npz"
BOUNDS_PATH = SAVED_MODELS_DIR / "norm_bounds.json"

MODEL_NAMES = ["isolation_forest", "autoencoder", "gmm"]


def _binary_metrics(name: str, scores: np.ndarray, y_true: np.ndarray) -> dict:
    """scores: normalized [0,1], 1.0 = most anomalous. y_true: 0=benign, 1=attack."""
    y_pred = (scores >= LOW_MEDIUM_THRESHOLD).astype(int)

    return {
        "name": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "scores": scores,
        "y_pred": y_pred,
    }


def _print_metrics_table(results: list[dict]) -> None:
    header = f"{'Model':<20}{'ROC-AUC':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'F2':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<20}{r['roc_auc']:>10.4f}{r['precision']:>12.4f}"
              f"{r['recall']:>10.4f}{r['f1']:>10.4f}{r['f2']:>10.4f}")


def _threshold_sweep(name: str, scores: np.ndarray, y_true: np.ndarray) -> None:
    print(f"\n{name}: recall/precision at candidate thresholds "
          "(current LOW_MEDIUM_THRESHOLD marked with *)")
    print(f"{'threshold':>10}{'precision':>12}{'recall':>10}{'f1':>10}{'f2':>10}")
    candidates = sorted(set([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                              LOW_MEDIUM_THRESHOLD, 0.5, 0.6, 0.7]))
    for t in candidates:
        y_pred = (scores >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        marker = " *" if t == LOW_MEDIUM_THRESHOLD else ""
        print(f"{t:>10.2f}{p:>12.4f}{r:>10.4f}{f1:>10.4f}{f2:>10.4f}{marker}")


def _print_confusion(name: str, cm: np.ndarray) -> None:
    print(f"\n{name} confusion matrix:")
    print("                  Predicted Benign   Predicted Attack")
    print(f"Actual Benign     {cm[0][0]:>16,}   {cm[0][1]:>16,}")
    print(f"Actual Attack     {cm[1][0]:>16,}   {cm[1][1]:>16,}")


def _per_attack_category_recall(scores: np.ndarray, attack_category: np.ndarray,
                                  threshold: float, y_true: np.ndarray) -> None:
    y_pred = (scores >= threshold).astype(int)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"\nEnsemble recall by attack category @ threshold={threshold} "
          f"(overall recall={overall_recall:.4f})")
    print("(BenignTraffic excluded -- recall isn't defined for a category with")
    print("no positives to catch; sorted worst-detected first)")
    categories = sorted(c for c in np.unique(attack_category) if c != BENIGN_LABEL)
    rows = []
    for cat in categories:
        mask = attack_category == cat
        n = int(mask.sum())
        caught = int(y_pred[mask].sum())
        rows.append((cat, n, caught, caught / n if n else 0.0))
    rows.sort(key=lambda r: r[3])
    print(f"{'Attack category':<30}{'n':>8}{'caught':>8}{'recall':>10}")
    for cat, n, caught, rec in rows:
        print(f"{cat:<30}{n:>8,}{caught:>8,}{rec:>10.4f}")
    n_zero = sum(1 for _, _, _, rec in rows if rec < 0.01)
    print(f"\n  -> {n_zero}/{len(rows)} categories still under 1% recall at this threshold")


def main():
    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"{SCORES_PATH} missing -- run ml/export_validation_scores.py first."
        )
    if not BOUNDS_PATH.exists():
        raise FileNotFoundError(
            f"{BOUNDS_PATH} missing -- run ml/fit_score_normalizer.py first."
        )

    data = np.load(SCORES_PATH, allow_pickle=True)
    attack_category = data["y_val"]
    y_true = np.where(attack_category == BENIGN_LABEL, 0, 1)

    n_benign = int((y_true == 0).sum())
    print(f"Validation set: {len(y_true):,} rows "
          f"(benign={n_benign:,}, attack={int((y_true == 1).sum()):,}, "
          f"{100 * n_benign / len(y_true):.2f}% benign)")
    print("NOTE: benign is the minority class here -- ROC-AUC is used below "
          "instead of PR-AUC for exactly that reason (see module docstring, point 1).\n")

    normalizer = ScoreNormalizer.load(BOUNDS_PATH)
    raw_scores = {name: data[name] for name in MODEL_NAMES}
    normalized = normalizer.transform(raw_scores)  # dict[str, np.ndarray], same shape as input

    # ---- Individual models ----
    individual_results = [
        _binary_metrics(name, normalized[name], y_true) for name in MODEL_NAMES
    ]

    # ---- Fused ensemble ----
    # fuse() is written per-flow (dict of floats -- see risk_engine.py's
    # docstring), so this loops row-by-row. Fine here: this runs once,
    # offline, over a few hundred thousand validation rows -- it is NOT
    # the live per-flow hot path, where that would matter.
    n = len(y_true)
    fused_scores = np.array([
        fuse({name: normalized[name][i] for name in MODEL_NAMES})
        for i in range(n)
    ])
    ensemble_result = _binary_metrics("ensemble (fused)", fused_scores, y_true)

    all_results = individual_results + [ensemble_result]

    print("=" * 72)
    print("Individual models vs. fused ensemble -- validation set")
    print("=" * 72)
    _print_metrics_table(all_results)

    for r in all_results:
        _print_confusion(r["name"], r["confusion_matrix"])

    _per_attack_category_recall(ensemble_result["scores"], attack_category,
                                 LOW_MEDIUM_THRESHOLD, y_true)

    _threshold_sweep("Ensemble (fused)", ensemble_result["scores"], y_true)

    print(
        f"\nReminder: LOW_MEDIUM_THRESHOLD={LOW_MEDIUM_THRESHOLD} is calibrated "
        "from real data (docs/DECISIONS.md #12). MEDIUM_HIGH_THRESHOLD and "
        "MODEL_WEIGHTS in ensemble/risk_engine.py are still placeholders -- "
        "see the TODOs there. Rerun this file after either is retuned, and "
        "again after any model is retrained."
    )


if __name__ == "__main__":
    main()