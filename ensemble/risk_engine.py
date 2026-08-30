"""
ensemble/risk_engine.py

Fuses the three normalized, oriented scores from score_normalization.py
into one final verdict per flow: a fused [0,1] score and a Low/Medium/
High label.

MODEL_WEIGHTS is still an equal-weight placeholder -- see its comment
below. LOW_MEDIUM_THRESHOLD is no longer a placeholder as of
2026-08-30: it's been calibrated against real validation-set data (see
docs/DECISIONS.md #12). MEDIUM_HIGH_THRESHOLD is still unvalidated.
"""

from shared.schemas import ModelName, RiskEvent, RiskLevel

# TODO(AGL): equal weighting is a starting default, not a tuned choice.
# The first real evaluation (docs/DECISIONS.md #12) found Isolation
# Forest alone recalled far more attacks than the equally-weighted
# fused ensemble at the OLD threshold (0.4) -- but that comparison was
# threshold-dependent and hasn't been re-checked at the new
# LOW_MEDIUM_THRESHOLD=0.05 below. Don't reweight from the old numbers;
# rerun ensemble/evaluate.py's per-model threshold sweep at 0.05 first,
# since AE/GMM's apparent weakness may have been an artifact of the
# same miscalibrated cutoff that was just fixed for the fused score,
# not a real gap in their standalone detection ability. Log whatever
# is decided in docs/DECISIONS.md -- "started equal, then tuned based
# on X" is a real answer; unexplained numbers aren't.
MODEL_WEIGHTS: dict[ModelName, float] = {
    "isolation_forest": 1 / 3,
    "autoencoder": 1 / 3,
    "gmm": 1 / 3,
}

# LOW_MEDIUM_THRESHOLD: CALIBRATED from real validation-set data -- see
# docs/DECISIONS.md #12 for the full reasoning and evidence table.
# Was 0.4 (an unvalidated placeholder). ensemble/evaluate.py's
# threshold-sweep on the real fused ensemble score showed 0.4 was
# missing ~74% of real attacks (recall=0.2570) despite a high ROC-AUC --
# the models WERE separating attack from benign scores, the cutoff was
# just drawn in the wrong place. 0.05 was the best-F2 candidate tested
# (F2 weights recall over precision, per DECISIONS.md #1: a missed
# attack costs more than a false alarm here): precision=0.9905,
# recall=0.9940, F2=0.9933 on the real validation set (run: 2026-08-30).
# Only 0.05-0.40 in steps of 0.05 were tested -- a finer search just
# below 0.05 hasn't been done and might help marginally, but returns
# are almost certainly diminishing given F2 is already 0.9933.
LOW_MEDIUM_THRESHOLD = 0.05

# TODO(AGL): still a placeholder, NOT yet backed by data -- do not
# treat this as validated just because LOW_MEDIUM_THRESHOLD above was
# checked and replaced. The Low/Medium boundary was tunable using
# binary attack-vs-benign recall/precision (an evaluate.py threshold
# sweep answers it directly). Medium-vs-High is a different question --
# "how much MORE anomalous than 'flagged' does something need to be to
# count as High" -- that recall/precision against a binary ground truth
# can't answer, since both Medium and High are "correctly flagged
# attack," just at different confidence. Needs the fused score's actual
# percentile distribution among already-flagged (attack) rows (e.g.
# "High = top 5% of flagged scores") before this number means anything.
MEDIUM_HIGH_THRESHOLD = 0.7


def fuse(normalized_scores: dict[str, float]) -> float:
    """
    Weighted average of already-normalized (0-1, 1=anomalous) scores.
    Divides by the weight of models actually present, not a hardcoded
    3 -- so if one model's score is ever missing (e.g. a model fails to
    load), this still returns a properly-scaled result using whichever
    models ARE present, instead of silently under-counting.
    """
    total_weight = sum(MODEL_WEIGHTS.get(name, 0.0) for name in normalized_scores)
    if total_weight == 0:
        raise ValueError(
            f"None of {list(normalized_scores)} are recognized in MODEL_WEIGHTS"
        )
    weighted_sum = sum(
        normalized_scores[name] * MODEL_WEIGHTS.get(name, 0.0)
        for name in normalized_scores
    )
    return weighted_sum / total_weight


def classify(final_score: float) -> RiskLevel:
    if final_score >= MEDIUM_HIGH_THRESHOLD:
        return "High"
    if final_score >= LOW_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def build_risk_event(
    flow_id: str,
    timestamp: float,
    normalized_scores: dict[str, float],
    protocol: str | None = None,
    src_ip: str | None = None,
    dst_ip: str | None = None,
    attack_category: str | None = None,
) -> RiskEvent:
    """The one function backend/ actually calls per flow."""
    final_score = fuse(normalized_scores)
    return RiskEvent(
        flow_id=flow_id,
        timestamp=timestamp,
        risk_level=classify(final_score),
        final_score=round(final_score, 4),
        model_breakdown=normalized_scores,
        protocol=protocol,
        src_ip=src_ip,
        dst_ip=dst_ip,
        attack_category=attack_category,
    )