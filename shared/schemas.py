"""
Shared data contracts for the IDS project.

Every piece of data that crosses a module boundary -- a model's score
going to the risk engine, a fused result going to the dashboard --
should be built using one of the classes below instead of a hand-built
dict. That's what lets four people's code stay compatible without
constantly comparing notes.

This is a first draft, written to unblock backend work. Review it with
your AGL before treating it as final -- once ensemble/ and frontend/
start importing it, changing a field name becomes a team-wide event,
not a one-line edit.
"""

from typing import Literal

from pydantic import BaseModel, Field

# The three models in the ensemble. A Literal (rather than a plain str)
# means a typo like "isolaton_forest" gets rejected by Pydantic at parse
# time instead of silently becoming an unrecognized fourth model.
ModelName = Literal["isolation_forest", "autoencoder", "lstm"]

RiskLevel = Literal["Low", "Medium", "High"]


class ModelOutput(BaseModel):
    """
    What a single model produces for a single flow.

    This stays inside the Python process -- it's how AGL's risk engine
    receives scores from the three models -- it does not itself get
    sent over the WebSocket.
    """

    model_name: ModelName
    flow_id: str
    raw_score: float
    # NOT yet normalized to a common scale. In particular, scikit-learn's
    # IsolationForest convention runs the opposite direction from the
    # other two models (lower raw score = more anomalous). AGL's
    # score_normalization step is responsible for fixing that before
    # fusion -- don't flip the sign here, or it'll get flipped twice.
    timestamp: float


class RiskEvent(BaseModel):
    """
    One fused detection result for a single flow.

    This is the exact shape that gets serialized to JSON and pushed over
    the WebSocket -- it IS the contract between backend and frontend.
    Member B's React code should be written against these exact field
    names.
    """

    flow_id: str
    src_ip: str
    dst_ip: str
    protocol: str  # kept as a plain string for now -- we'll tighten this
    # to a Literal once Member A's EDA confirms the exact
    # protocol value formats in CICIoT2023.
    timestamp: float
    risk_level: RiskLevel
    final_score: float = Field(ge=0.0, le=1.0)
    model_breakdown: dict[str, float]
    # e.g. {"isolation_forest": 0.82, "autoencoder": 0.41, "lstm": 0.77}
    # -- normalized 0-1 scores, one per model, so the dashboard can show
    # which model(s) drove a given risk level.