"""
shared/schemas.py

The data contracts every module in this project codes against.

Why this file exists
---------------------
Four people are building four modules (ml/, ensemble/, backend/, frontend/)
that hand data to each other. If everyone freelances the shape of that data,
the mismatch shows up at integration time -- the worst possible time to
find it, since by then everyone's code already "works" in isolation.

Pydantic gives us two things for free:
1. One obvious place to look up "what fields does X actually have".
2. Runtime validation -- if some code tries to build a RiskEvent with
   risk_level="high" (lowercase) instead of "High", it fails immediately
   and loudly instead of quietly sending malformed JSON to the dashboard.

No detection logic lives in this file on purpose. Just shapes.

Reconciliation note: this replaces the version currently committed, which
predates the LSTM->GMM swap (docs/DECISIONS.md #6-9) and had src_ip/dst_ip
as REQUIRED fields. CICIoT2023's CSV has no IP column at all (DECISIONS.md
#6), so replay_engine.py could never actually have populated those --
made them Optional instead of dropping them; see the field comments below.
"""

from pydantic import BaseModel, Field
from typing import Literal

# The three models in the ensemble, defined once so every other file
# (score_normalization.py, risk_engine.py, evaluate.py, backend/...) refers
# to the same three strings instead of retyping them.
ModelName = Literal["isolation_forest", "autoencoder", "gmm"]

RiskLevel = Literal["Low", "Medium", "High"]


class ModelOutput(BaseModel):
    """
    One model's raw anomaly score for one flow.

    Produced by:
      - ml/ during offline evaluation (evaluate.py loads the saved models
        and scores the held-out test set)
      - backend/ during live inference, after loading the frozen models
    Consumed by:
      - ensemble/score_normalization.py
    """
    model_name: ModelName
    flow_id: str
    raw_score: float
    # NOT yet normalized to a common scale, and NOT sign-corrected --
    # isolation_forest and gmm both run "lower = more anomalous", opposite
    # to autoencoder. ensemble/score_normalization.py owns fixing both.
    # Don't flip the sign here, or it gets flipped twice.
    timestamp: float


class RiskEvent(BaseModel):
    """
    The final fused verdict for one flow -- what actually reaches the
    dashboard.

    Produced by:
      - ensemble/risk_engine.py
    Consumed by:
      - backend/websocket_manager.py (serializes this to JSON, pushes it)
      - frontend/ (renders the risk gauge, alert feed, trend chart from it)
    """
    flow_id: str

    # Optional, not required: CICIoT2023's CSV has no source/destination IP
    # column at all (docs/DECISIONS.md #6), so replay_engine.py has nothing
    # real to put here. Leave None for an honest baseline, or have
    # replay_engine.py generate a clearly-synthetic placeholder (e.g. a
    # per-flow pseudo-IP derived from flow_id) if the dashboard wants
    # something to display -- GL's call, since replay_engine.py is where
    # this gets populated. Either way: never present these as real
    # captured addresses in the report or viva.
    src_ip: str | None = None
    dst_ip: str | None = None

    # Also not guaranteed -- only populate if CICIoT2023's feature set turns
    # out to include a reconstructable protocol indicator (check
    # ml/saved_models/feature_columns.json once preprocessing.py has run).
    # Optional for the same reason as src_ip/dst_ip: don't block RiskEvent
    # construction on data that might not exist.
    protocol: str | None = None

    timestamp: float
    risk_level: RiskLevel
    final_score: float = Field(ge=0.0, le=1.0)
    model_breakdown: dict[str, float]
    # e.g. {"isolation_forest": 0.82, "autoencoder": 0.41, "gmm": 0.77}
    # -- normalized 0-1 scores, one per model, so the dashboard can show
    # which model(s) drove a given risk level.

    # Ground-truth attack category from the CICIoT2023 row being replayed
    # (e.g. "DDoS-SYN_Flood", "Benign"), shown on the dashboard for demo
    # context ONLY. NOT a model prediction -- all three models are trained
    # benign-only/unsupervised (DECISIONS.md #1), so none of them classify
    # attack type. Populated only because we control the replay data and
    # already know the label; would be None for genuinely live traffic.
    attack_category: str | None = None