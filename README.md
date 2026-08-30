# Dynamic Adaptive IDS: Ensemble Anomaly Detection with Live Threat Visualization

A machine-learning intrusion detection system that fuses three unsupervised anomaly detectors —
Isolation Forest, Autoencoder, and Gaussian Mixture Model (GMM) — into a single weighted-ensemble
risk score, streamed to a live dashboard over WebSocket. Built on the CICIoT2023 dataset.

This is a Sem 1 baseline: replay of held-out dataset rows over a live pipeline, **not** live
network packet capture. See `docs/DECISIONS.md` #3 for why, and never describe it otherwise in
the report or viva.

## Team

| Role | Module | Owns |
|---|---|---|
| Jatin | `backend/` | FastAPI app, WebSocket manager, replay engine, integration |
| Pranav | `ensemble/` | Score normalization, weighted fusion, risk thresholds, ensemble evaluation |
| Sanchit | `ml/` | Preprocessing, EDA, training all three models, per-model metrics |
| Sangram | `frontend/` | React dashboard, WebSocket client, risk visualization |

Full work breakdown and dependencies: `docs/ARCHITECTURE.md` section 6.

## How it works, in one paragraph

All three models train **benign-only** (unsupervised novelty detection — see `docs/DECISIONS.md`
#1 for why this matters for the "detects previously-unseen attacks" claim). Each produces a raw
anomaly score in its own scale and direction; `ensemble/score_normalization.py` sign-corrects and
rescales all three to `[0,1]` where `1.0` always means "more anomalous," using bounds fit once on
the validation set and frozen afterward. `ensemble/risk_engine.py` combines the three normalized
scores into one fused score by weighted average, then classifies it into Low / Medium / High using
two thresholds. `backend/replay_engine.py` streams held-out validation rows through this whole
pipeline at a fixed interval and broadcasts the result to any connected dashboard over `/ws`.

Full architecture, including why GMM replaced the originally-planned LSTM and the exact
sign-flip/normalization math: `docs/ARCHITECTURE.md`. Full decision history, with evidence, for
every non-obvious choice in this repo: `docs/DECISIONS.md`.

## Setup

Requires **Python 3.10 or 3.11** — `requirements.txt` pins `tensorflow==2.15.0` and
`numpy==1.26.4`, neither of which has installable wheels for 3.13+.

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

Place the CICIoT2023 CSVs at:
```
data/CICIOT23/train/train.csv
data/CICIOT23/test/test.csv
data/CICIOT23/validation/validation.csv
```

## Running the full pipeline

Run these in order. Everything under `ml/` and `ensemble/evaluate.py` works whether invoked from
the project root or from inside `ml/` — Python resolves same-directory imports either way, so
don't worry about `cd`ing around; `ensemble/evaluate.py` specifically should be run from the
project root since it imports across the `ml/` and `ensemble/` packages.

```bash
# 1. Preprocess the raw CSVs -- benign-only training matrix, deduplicated test/validation splits
python ml/preprocessing.py

# 2. Verify nothing got truncated/corrupted before doing anything expensive with it
python ml/verify_processed_data.py
# must print "ALL CHECKS PASSED" before continuing

# 3. Train all three models (each loads data/processed/X_train_benign.npy directly)
python ml/train_isolation_forest.py
python ml/train_autoencoder.py
python ml/train_gmm.py

# 4. Score the validation set with the trained models, then fit + freeze the normalizer
python ml/export_validation_scores.py
python ml/fit_score_normalizer.py

# 5. Run the ensemble evaluation -- individual models vs. fused ensemble, per-attack-category
#    recall, and a threshold-sweep table. Run from the project root:
python ensemble/evaluate.py

# 6. Run the backend (serves /health and the live /ws stream)
uvicorn backend.main:app --reload
```

If step 1 or 2 fails partway (interrupted write, disk space, etc.), `verify_processed_data.py`
will tell you exactly which file is bad — rerun `preprocessing.py` rather than guessing.

## Current status

- **Preprocessing** — done, verified against the real dataset (integrity-checked, all shapes and
  row counts match `docs/ARCHITECTURE.md`'s EDA numbers exactly).
- **Training** — done. All three models trained and saved successfully.
- **Ensemble evaluation** — done. First real numbers produced; `LOW_MEDIUM_THRESHOLD` has been
  calibrated from real validation data (was an unvalidated 0.4 placeholder, now 0.05 — see
  `docs/DECISIONS.md` #12 for the full evidence and reasoning).
- **Backend** — `main.py`/`websocket_manager.py`/`replay_engine.py` wiring verified: a connected
  WebSocket client actually receives live broadcasts end-to-end.
- **Still open:**
  - `MEDIUM_HIGH_THRESHOLD` and `MODEL_WEIGHTS` are still placeholders, not yet calibrated from
    data — see `docs/DECISIONS.md` #12's "next steps" for what that needs.
  - Frontend (`frontend/`) — not started.
  - `backend/model_runtime.py` — exists and loads/scores the real trained models, but hasn't been
    independently reviewed alongside the rest of the backend wiring.

## Repo structure

See `docs/ARCHITECTURE.md` section 5 for the full annotated tree. Quick orientation:
- `ml/` — everything that touches raw data or trains a model
- `ensemble/` — everything that turns three raw scores into one Low/Medium/High verdict
- `backend/` — FastAPI + WebSocket serving layer
- `shared/schemas.py` — the Pydantic contract every module codes against; check here first if
  you're unsure what fields something should have
- `docs/` — `ARCHITECTURE.md` (how it works), `DECISIONS.md` (why it works that way, with
  evidence), `API_CONTRACT.md` (WebSocket/REST contract for the frontend)
- `tests/` — run with `PYTHONPATH=. pytest tests/` from the project root

## A note on scope

This is explicitly a **Sem 1 baseline**. Concept drift/retraining, SHAP-based explainability,
live packet capture, and cross-dataset validation are all Future Scope, not required for this
milestone — see the project synopsis and `docs/ARCHITECTURE.md` section 1 for what's actually in
scope right now.
