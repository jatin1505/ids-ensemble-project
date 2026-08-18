## 1. System overview — two pipelines
 
The system is two pipelines that share code and models but never run at the same time:
 
1. **Offline — Training & Evaluation.** Runs once (per retraining), on historical CICIoT2023
   data. Produces trained model files + performance numbers. Never touches "live" traffic.
2. **Online — Live Inference.** Runs continuously once the system is switched on. Loads the
   already-trained, *frozen* models and scores new flows — here, replayed from held-out data —
   in real time, streaming results to the dashboard.
Nothing learns while the demo is running. Models are frozen the moment training finishes.
 
**Baseline = dataset replay, not live capture.** The system reads held-out CICIoT2023 rows and
streams them through the live pipeline at a controlled rate over WebSocket. This must never be
described as live network monitoring in the report or viva. Live packet capture is Future Scope
(Phase 3, only if time allows).
 
---
 
## 2. Models — Isolation Forest, Autoencoder, GMM
 
All three models are trained **benign-only** (unsupervised / novelty detection). Attack labels
are held out entirely until evaluation. This is required, not optional — see `DECISIONS.md` #1.
 
| Model | Question it asks | Trained on | Raw output | Direction |
|---|---|---|---|---|
| Isolation Forest | "Is this point easy to isolate?" | Benign-only matrix | `decision_function()` (avg. path length) | Higher = more normal |
| Autoencoder | "Can I reconstruct this point?" | Benign-only matrix | Reconstruction MSE | Higher = more anomalous |
| Gaussian Mixture Model (GMM) | "How likely is this point under the distribution of normal traffic?" | Benign-only matrix | `score_samples()` (log-likelihood) | Higher = more normal |
 
GMM replaced the originally planned LSTM. See `DECISIONS.md` #6 for the full reasoning.
 
All three consume the **same flat, scaled feature matrix** — no special sequence/window
preprocessing is needed for any of them (this was previously only true for IF and AE; dropping
LSTM removed the one component that needed different input shape).
 
---
 
## 3. Ensemble — score normalization, fusion, thresholding
 
**Step 1 — Sign correction (per model, per row).**
Two of three raw scores run "higher = normal," one runs "higher = anomalous." Both IF and GMM
must be sign-flipped so all three agree: **higher = more anomalous**.
 
```
anomaly_score_if  = -decision_function(x)      # flip
anomaly_score_ae  =  reconstruction_mse(x)      # no flip
anomaly_score_gmm = -log_likelihood(x)          # flip
```
 
> Correction from the original plan: only IF was flagged for flipping there, because the
> original third model (LSTM) also produced "higher = anomalous" prediction error, like AE.
> GMM does not — its raw direction matches IF's. `ensemble/score_normalization.py` must flip
> **both** IF and GMM.
 
**Step 2 — Normalize to [0,1].** Fit a scaler (min-max or percentile-based, clipped against
outliers) per model on the **validation set's** sign-corrected scores. Never refit at inference.
 
**Step 3 — Weighted fusion.**
```
final_score = w_if * norm_if + w_ae * norm_ae + w_gmm * norm_gmm     (weights sum to 1)
```
Baseline: equal weights (1/3 each). Tuned weights (grid search against validation F2) are a
reportable ablation, not the default — avoid overfitting weights to a small validation set.
 
**Step 4 — Threshold to Low / Medium / High.** Two thresholds (`t_low`, `t_high`) derived from
the validation fused-score distribution, not recomputed live.
 
**Step 5 — Evaluation.** `ensemble/evaluate.py` computes F1, F2, PR-AUC, and confusion matrices
per individual model AND for the fusion, broken down per attack category. This is where
Objective 3 (ensemble vs. individual comparison) gets its numbers. Runs at development time only.
 
---
 
## 4. Dataset
 
**Source:** CICIoT2023, Kaggle pre-split mirror (`CICIOT23/{train,test,validation}.csv`).
Confirmed to use the same 46-feature flow-window-aggregated schema as the official UNB release
— no IP, timestamp, or flow/session ID in either version (see `DECISIONS.md` #6).
 
| Split | Rows | Benign | Benign % | Attack classes |
|---|---|---|---|---|
| train | 5,491,971 | 129,538 | 2.36% | 34 total (33 attack + benign) |
| test | 1,176,851 | 27,709 | 2.35% | 34 total |
| validation | 1,176,851 | 27,519 | 2.34% | 34 total |
 
**EDA findings:**
- No NaNs, no infinities, no negative values across all 46 numeric columns (this pre-split
  mirror already handles the ratio-feature divide-by-zero issue present in the raw UNB export).
- **Cross-split exact-duplicate leakage**: ~4.9% of test rows and ~4.9% of validation rows are
  exact duplicates of rows in train (57,847 and 57,744 rows respectively; 12,422 rows overlap
  between test and validation). `ml/preprocessing.py` must drop these before use — hash every
  row, discard test/validation rows whose hash appears in train. Log the dropped count.
- No per-source IP or timestamp exists in this feature format, in any distribution of
  CICIoT2023 (Kaggle or official UNB CSV) — the CSVs are generated by summarizing over
  fixed-size packet windows during PCAP→CSV conversion, which strips that information by
  design. Reconstructing it would require parsing raw PCAPs, which is out of scope (see
  `DECISIONS.md` #6/#7).
**Dataset choice reaffirmed** against current alternatives (TII-SSRC-23, Edge-IIoTset,
Gotham Dataset 2025) — CICIoT2023 kept. See `DECISIONS.md` #8.
 
---
 
## 5. Repo structure
 
```
ids-ensemble-project/
├── data/                     # gitignored — raw & processed CICIoT2023 subsets
├── ml/                        # Member A
│   ├── preprocessing.py       # clean, dedupe cross-split rows, scale, benign/attack split
│   ├── train_isolation_forest.py
│   ├── train_autoencoder.py
│   ├── train_gmm.py           # replaces train_lstm.py
│   ├── config.py               # shared seeds, paths, feature list
│   └── saved_models/
├── ensemble/                    # AGL
│   ├── score_normalization.py  # sign-flip IF + GMM, min-max/percentile scale
│   ├── risk_engine.py          # weighted fusion + Low/Med/High thresholds
│   └── evaluate.py             # per-model + ensemble metrics, per attack category
├── backend/                     # GL
│   ├── main.py
│   ├── websocket_manager.py
│   └── replay_engine.py        # streams held-out rows; no per-source buffering needed
├── frontend/                    # Member B
│   └── src/...
├── shared/
│   └── schemas.py               # Pydantic models — the contract everyone codes against
├── docs/
│   ├── ARCHITECTURE.md          # this file
│   ├── API_CONTRACT.md
│   └── DECISIONS.md
├── tests/
├── requirements.txt
└── README.md
```
 
**Removed from the original plan:** `ml/sequence_builder.py` (group-by-IP / sort-by-timestamp
windowing for LSTM) — no longer needed, since the dataset can't support it and GMM doesn't
require sequence input.
 
---
 
## 6. Work distribution (unchanged mapping, lighter ml/ workload)
 
| Role | Module | Core work | Blocked on |
|---|---|---|---|
| Member A | `ml/` | EDA, cleaning, cross-split dedup, benign/attack split, training & tuning IF/AE/GMM, per-model metrics, model artifacts | `shared/schemas.py` |
| AGL | `ensemble/` | Score normalization (IF + GMM sign-flip), weighted fusion, threshold selection, ensemble-vs-individual evaluation | Model output schema; real scores from Member A |
| GL | `backend/` | FastAPI app, WebSocket manager, replay engine, wiring risk engine output to clients, integration | WebSocket schema; real risk events from AGL |
| Member B | `frontend/` | React dashboard, WebSocket client, risk gauge, live alert feed, per-model breakdown view (IF/AE/GMM, not IF/AE/LSTM) | WebSocket schema |
 
Dropping LSTM reduces Member A's workload (no sequence builder, no LSTM training
instability/tuning) — that slack is better spent on deeper evaluation (weight-tuning ablation,
per-attack-category breakdown) than on new scope.
