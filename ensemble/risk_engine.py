"""
ensemble/risk_engine.py

Fuses the three normalized, oriented scores from score_normalization.py
into one final verdict per flow: a fused [0,1] score and a Low/Medium/
High label.

THIS FILE WAS EMPTY in the repo -- score_normalization.py and its tests
are AGL's verified work, but this one and evaluate.py hadn't been
written yet. This is a first draft to unblock backend integration, not
a finished, reviewed module. Two things below are explicitly
placeholders, not tuned decisions -- see the TODO comments on
MODEL_WEIGHTS and the two thresholds. AGL should review this whole file
before it's treated as final.
"""

from shared.schemas import ModelName, RiskEvent, RiskLevel

# TODO(AGL): equal weighting is a starting default, not a tuned choice.
# Once evaluate.py can show each model's standalone precision/recall on
# the validation set, consider weighting toward whichever model(s)
# separate benign from attack traffic best on their own. Either way,
# log the reasoning in docs/DECISIONS.md -- "started equal, then tuned
# based on X" is a real answer; unexplained numbers aren't.
MODEL_WEIGHTS: dict[ModelName, float] = {
    "isolation_forest": 1 / 3,
    "autoencoder": 1 / 3,
    "gmm": 1 / 3,
}

# TODO(AGL): placeholders, not calibrated. Should come from where the
# FUSED score actually lands on the validation set once
# validation_raw_scores.npz exists -- e.g. picking cutoffs that hit a
# target recall on known attacks (weight toward recall, per
# docs/DECISIONS.md -- a missed attack costs more than a false alarm
# here). Not percentiles of any single model's raw score -- percentiles
# of the fused score, after weighting.
LOW_MEDIUM_THRESHOLD = 0.4
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