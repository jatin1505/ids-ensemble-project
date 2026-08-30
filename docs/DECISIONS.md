### 1. Benign-only training for all three models (not a supervised/unsupervised mix)
CICIoT2023 ships attack labels, which makes it tempting to train the sequence model as a
supervised classifier — it would likely score higher on paper. Rejected: if any one model is
trained on labeled attack types, its contribution is only valid for attack types it already saw,
which directly contradicts the "detects known and previously unseen, zero-day-style attacks"
claim (Expected Outcome #1). That claim only holds for the whole ensemble if every model learned
"normal" rather than memorizing "attack." All labels are held out until evaluation.
 
### 2. Weighted score fusion, not discrete vote-counting
Isolation Forest / Autoencoder / GMM don't output comparable numbers out of the box, and their
raw directions disagree (see #9 below). Normalizing each to [0,1] and combining via weighted sum
preserves confidence information that discrete voting throws away — the standard approach in
anomaly-ensemble literature. A learned meta-model (logistic regression over the three scores) is
a reasonable future-work mention, not the baseline.
 ### 1. Benign-only training for all three models (not a supervised/unsupervised mix)
CICIoT2023 ships attack labels, which makes it tempting to train the sequence model as a
supervised classifier — it would likely score higher on paper. Rejected: if any one model is
trained on labeled attack types, its contribution is only valid for attack types it already saw,
which directly contradicts the "detects known and previously unseen, zero-day-style attacks"
claim (Expected Outcome #1). That claim only holds for the whole ensemble if every model learned
"normal" rather than memorizing "attack." All labels are held out until evaluation.
 
### 2. Weighted score fusion, not discrete vote-counting
Isolation Forest / Autoencoder / GMM don't output comparable numbers out of the box, and their
raw directions disagree (see #9 below). Normalizing each to [0,1] and combining via weighted sum
preserves confidence information that discrete voting throws away — the standard approach in
anomaly-ensemble literature. A learned meta-model (logistic regression over the three scores) is
a reasonable future-work mention, not the baseline.
 
### 3. Dataset replay, not live capture, for the baseline
FF180's Figure 1 shows CICIoT2023 and Live Capture as parallel input options, but the synopsis's
own Future Scope section frames live packet capture as what "moves beyond dataset replay toward
true real-time detection" — meaning replay *is* the Sem 1 baseline. The system reads held-out
CICIoT2023 rows and streams them over WebSocket at a controlled rate. Never describe this as live
network monitoring in the report or viva.
 
### 4. "Model-attributable," not "explainable," for the baseline
Concept drift/retraining and SHAP-based explainability are both explicitly Future Scope in FF180.
The baseline dashboard shows *which models* flagged a flow and how strongly (model-level
interpretability), not *which features* drove the decision (that's SHAP). Reserve
"explainable/XAI" for if SHAP is actually added later.
 
### 5. Cross-split duplicate rows found — dedup before use
Hashed every row across the Kaggle CICIoT2023 train/test/validation split. Found ~4.9% of test
rows (57,847) and ~4.9% of validation rows (57,744) are exact duplicates of rows in train; 12,422
rows overlap between test and validation. The split was done by random row shuffling, not
dedup-then-split. Since training is benign-only, this doesn't leak attack information into
training — but it does mean some evaluation rows were literally seen during training, inflating
apparent benign-reconstruction performance. `ml/preprocessing.py` drops any test/validation row
whose hash matches a train row before evaluation.

**Confirmed against the real dataset (2026-08-30):** rerunning `ml/preprocessing.py` against the
actual CICIoT2023 CSVs reproduced these exact numbers — 57,847 test duplicates dropped, 57,744
validation duplicates dropped. Not just an estimate from earlier EDA; verified by a real run.
 
### 6. LSTM dropped — no IP/timestamp/session ID exists in CICIoT2023's CSV format
The original plan required grouping flows by source IP, sorting by timestamp, and building
sliding windows so the LSTM could catch escalating patterns (e.g. a ramping DDoS). Checked the
actual data: the CSV schema is 46 numeric features + label — no IP, timestamp, or flow/session ID
column, in either the Kaggle mirror or the official UNB release. This is not a Kaggle
re-uploader's mistake — CIC's own pipeline (Mergecap → PySpark → TCPDump → DPKT) summarizes
packets into fixed-size windows before export, and that summarization step is what strips
per-packet identity/timing metadata by design (confirmed against the original CICIoT2023 paper
and the official UNB dataset page). Row order in the CSV is also fully shuffled — 24 distinct
attack classes appeared within the first 1,000 rows sampled, so there's no fallback "use file
order as a proxy for time" option either. Without source/time information, genuine per-device
temporal sequences cannot be reconstructed from this data.
 
### 7. PCAP-based re-extraction investigated and rejected as infeasible
The raw IP/timestamp information does exist, but only in CICIoT2023's PCAP files, which would
require re-implementing CIC's own feature-extraction pipeline (packet parsing, flow
reconstruction, statistical feature engineering) from scratch. Uncompressed PCAPs for this
dataset run into the hundreds of GB — not viable on student hardware — and the effort required
is comparable to a separate project, not an extension of this one. Rejected for this timeline.
 
### 8. LSTM replaced with Gaussian Mixture Model (GMM)
Evaluated three sequence-free alternatives — GMM, One-Class SVM, Local Outlier Factor (novelty
mode) — against: distinctness from IF (tree-based isolation) and AE (neural reconstruction),
training time on ~130K benign rows, per-row inference latency (matters directly for the
WebSocket real-time scoring loop), and explainability for the viva.
 
- **LOF** rejected: novelty-mode inference requires a k-NN lookup against the training set on
  every scored row — directly works against the real-time WebSocket design.
- **One-Class SVM** viable but weaker fit: RBF kernel training is roughly O(n²)–O(n³), so it
  needs subsampling to ~20-30K rows to stay practical, and results are sensitive to
  nu/gamma tuning — real, non-trivial extra work.
- **GMM** selected: a genuinely different detection angle (density estimation vs. isolation vs.
  reconstruction), fast to train on the full benign set (seconds–low minutes, no GPU needed —
  helps rather than hurts the student-hardware constraint), fast single-row inference (just
  evaluate the mixture density), and the cleanest three-sentence pitch for ensemble diversity in
  the viva: *"IF asks is this point easy to isolate, AE asks can I reconstruct this point, GMM
  asks how likely is this point under the distribution of normal traffic."*
Net effect: this also removed `ml/sequence_builder.py` entirely and removed the need for any
stateful per-source buffering in the backend/replay engine — a simplification, not just a swap.
 
### 9. Score-normalization correction: GMM needs the same sign flip as Isolation Forest
Originally only Isolation Forest was flagged as needing its raw score sign-flipped before fusion
(when the third model was LSTM, whose prediction error ran the same "higher = anomalous"
direction as the Autoencoder's reconstruction error). GMM's raw output (`score_samples()`,
log-likelihood) runs "higher = more normal," same direction as Isolation Forest, not the same
direction as the Autoencoder. `ensemble/score_normalization.py` must flip **both** IF and GMM
scores (`anomaly_score = -raw_score`) so all three agree on direction before normalization.
 
### 10. Dataset choice reaffirmed: staying on CICIoT2023
Checked current alternatives within a 5-6 year window: TII-SSRC-23 (2023), Edge-IIoTset (2022),
and Gotham Dataset 2025 (a newer IoT testbed that captures traffic separately per device — its
real advantage is that it *would* support genuine per-source sequences, unlike CICIoT2023).
Not switching: doing so would discard a completed EDA and a locked, validated model plan, and
Gotham would still require building a PCAP→CSV feature-extraction pipeline ourselves to get
anything beyond raw packets — the same scope problem decision #7 already ruled out. CICIoT2023
also remains the most widely cited IoT IDS benchmark from this period, which matters for the
report's related-work section. Noted as a genuine Future Scope item: "device-separated datasets
such as Gotham 2025 could enable real temporal sequence modeling in future work."
 
### 11. `ml/sequence_builder.py` removed from the repo
Direct consequence of #6/#8 — no model in the current design requires windowed/sequential input,
so the module has no purpose. All three models consume the same flat, scaled feature matrix.

### 12. LOW_MEDIUM_THRESHOLD recalibrated from real validation data (0.4 -> 0.05)
`ensemble/risk_engine.py`'s `LOW_MEDIUM_THRESHOLD` was a 0.4 placeholder (see #2/original TODO)
pending real fused-ensemble scores. Once `ml/export_validation_scores.py` and
`ml/fit_score_normalizer.py` produced real scores from the trained models, `ensemble/evaluate.py`
was extended with a threshold-sweep and confirmed the placeholder was badly miscalibrated, not
just imprecise.

**Evidence** (validation set: 1,119,107 rows, 26,176 benign / 1,092,931 attack, 2.34% benign; run
2026-08-30):

At the old threshold (0.4), the fused ensemble's per-attack-category recall was catastrophically
bimodal: 3 of 33 attack categories (the ICMP-flood family: DDoS-ICMP_Flood, DDoS-RSTFINFlood,
DDoS-ICMP_Fragmentation — together 25.7% of all attack rows) were caught at ~97-99.9% recall,
while every other category sat at 0.0000-0.03 recall, including large categories like
DDoS-UDP_Flood (129,846 rows, 8 caught). Overall recall was 0.2570 — almost exactly the ICMP-flood
share of total attacks, confirming the ensemble was essentially only detecting that one family.

A full threshold sweep on the fused score (0.05 to 0.70) showed this was a threshold problem, not
a model problem:

| threshold | precision | recall | F1 | F2 |
|---|---|---|---|---|
| 0.05 | 0.9905 | 0.9940 | 0.9922 | 0.9933 |
| 0.10 | 0.9945 | 0.9885 | 0.9915 | 0.9897 |
| 0.15 | 0.9966 | 0.7913 | 0.8822 | 0.8253 |
| 0.20 | 0.9977 | 0.5411 | 0.7016 | 0.5956 |
| 0.25 | 0.9980 | 0.3039 | 0.4659 | 0.3530 |
| 0.30 | 0.9990 | 0.2872 | 0.4461 | 0.3349 |
| 0.35 | 1.0000 | 0.2577 | 0.4098 | 0.3026 |
| 0.40 (old) | 1.0000 | 0.2570 | 0.4090 | 0.3019 |

At 0.05, precision and recall are simultaneously ~0.99 — not just a better aggregate number, but
confirmed via `ensemble/evaluate.py`'s per-category breakdown to hold across attack types, not
only the 3 ICMP-flood categories that were already easy at 0.4.

**Decision:** `LOW_MEDIUM_THRESHOLD` set to **0.05** — the best-F2 candidate tested. F2 (recall-
weighted) was used as the selection criterion per decision #1's own reasoning: a missed attack
costs more than a false alarm for this system. Only 0.05-0.40 in steps of 0.05 were swept; a finer
search just below 0.05 hasn't been done and might help marginally, but F2=0.9933 leaves little
room to improve.

**Explicitly NOT changed by this decision, still placeholders:**
- `MEDIUM_HIGH_THRESHOLD` (0.7) — Low/Medium is a binary attack-vs-benign question that a
  recall/precision sweep answers directly; Medium-vs-High is "how much MORE anomalous than
  'flagged' to count as High," which needs the fused score's percentile distribution among
  already-flagged rows, not binary recall/precision. That analysis hasn't been done.
- `MODEL_WEIGHTS` (equal 1/3 each) — the original evaluation run (at the old 0.4 threshold) showed
  Isolation Forest alone recalling far more than AE/GMM (0.7707 vs ~0.2565), which on its face
  argues for reweighting toward IF. Deliberately not acted on: that gap was measured at a threshold
  now known to be miscalibrated, so it may partly or fully be an artifact of the same problem this
  decision just fixed rather than a real gap in AE/GMM's standalone detection ability. Needs a
  per-model threshold sweep (not just the fused one) at the new threshold before reweighting.

**Next steps this decision opens up:** per-model (not just fused) threshold sweeps to inform
`MODEL_WEIGHTS`; a percentile-based analysis to calibrate `MEDIUM_HIGH_THRESHOLD`; rerunning both
after any model retraining, since these numbers are tied to this specific validation run.
### 3. Dataset replay, not live capture, for the baseline
FF180's Figure 1 shows CICIoT2023 and Live Capture as parallel input options, but the synopsis's
own Future Scope section frames live packet capture as what "moves beyond dataset replay toward
true real-time detection" — meaning replay *is* the Sem 1 baseline. The system reads held-out
CICIoT2023 rows and streams them over WebSocket at a controlled rate. Never describe this as live
network monitoring in the report or viva.
 
### 4. "Model-attributable," not "explainable," for the baseline
Concept drift/retraining and SHAP-based explainability are both explicitly Future Scope in FF180.
The baseline dashboard shows *which models* flagged a flow and how strongly (model-level
interpretability), not *which features* drove the decision (that's SHAP). Reserve
"explainable/XAI" for if SHAP is actually added later.
 
### 5. Cross-split duplicate rows found — dedup before use
Hashed every row across the Kaggle CICIoT2023 train/test/validation split. Found ~4.9% of test
rows (57,847) and ~4.9% of validation rows (57,744) are exact duplicates of rows in train; 12,422
rows overlap between test and validation. The split was done by random row shuffling, not
dedup-then-split. Since training is benign-only, this doesn't leak attack information into
training — but it does mean some evaluation rows were literally seen during training, inflating
apparent benign-reconstruction performance. `ml/preprocessing.py` drops any test/validation row
whose hash matches a train row before evaluation.
 
### 6. LSTM dropped — no IP/timestamp/session ID exists in CICIoT2023's CSV format
The original plan required grouping flows by source IP, sorting by timestamp, and building
sliding windows so the LSTM could catch escalating patterns (e.g. a ramping DDoS). Checked the
actual data: the CSV schema is 46 numeric features + label — no IP, timestamp, or flow/session ID
column, in either the Kaggle mirror or the official UNB release. This is not a Kaggle
re-uploader's mistake — CIC's own pipeline (Mergecap → PySpark → TCPDump → DPKT) summarizes
packets into fixed-size windows before export, and that summarization step is what strips
per-packet identity/timing metadata by design (confirmed against the original CICIoT2023 paper
and the official UNB dataset page). Row order in the CSV is also fully shuffled — 24 distinct
attack classes appeared within the first 1,000 rows sampled, so there's no fallback "use file
order as a proxy for time" option either. Without source/time information, genuine per-device
temporal sequences cannot be reconstructed from this data.
 
### 7. PCAP-based re-extraction investigated and rejected as infeasible
The raw IP/timestamp information does exist, but only in CICIoT2023's PCAP files, which would
require re-implementing CIC's own feature-extraction pipeline (packet parsing, flow
reconstruction, statistical feature engineering) from scratch. Uncompressed PCAPs for this
dataset run into the hundreds of GB — not viable on student hardware — and the effort required
is comparable to a separate project, not an extension of this one. Rejected for this timeline.
 
### 8. LSTM replaced with Gaussian Mixture Model (GMM)
Evaluated three sequence-free alternatives — GMM, One-Class SVM, Local Outlier Factor (novelty
mode) — against: distinctness from IF (tree-based isolation) and AE (neural reconstruction),
training time on ~130K benign rows, per-row inference latency (matters directly for the
WebSocket real-time scoring loop), and explainability for the viva.
 
- **LOF** rejected: novelty-mode inference requires a k-NN lookup against the training set on
  every scored row — directly works against the real-time WebSocket design.
- **One-Class SVM** viable but weaker fit: RBF kernel training is roughly O(n²)–O(n³), so it
  needs subsampling to ~20-30K rows to stay practical, and results are sensitive to
  nu/gamma tuning — real, non-trivial extra work.
- **GMM** selected: a genuinely different detection angle (density estimation vs. isolation vs.
  reconstruction), fast to train on the full benign set (seconds–low minutes, no GPU needed —
  helps rather than hurts the student-hardware constraint), fast single-row inference (just
  evaluate the mixture density), and the cleanest three-sentence pitch for ensemble diversity in
  the viva: *"IF asks is this point easy to isolate, AE asks can I reconstruct this point, GMM
  asks how likely is this point under the distribution of normal traffic."*
Net effect: this also removed `ml/sequence_builder.py` entirely and removed the need for any
stateful per-source buffering in the backend/replay engine — a simplification, not just a swap.
 
### 9. Score-normalization correction: GMM needs the same sign flip as Isolation Forest
Originally only Isolation Forest was flagged as needing its raw score sign-flipped before fusion
(when the third model was LSTM, whose prediction error ran the same "higher = anomalous"
direction as the Autoencoder's reconstruction error). GMM's raw output (`score_samples()`,
log-likelihood) runs "higher = more normal," same direction as Isolation Forest, not the same
direction as the Autoencoder. `ensemble/score_normalization.py` must flip **both** IF and GMM
scores (`anomaly_score = -raw_score`) so all three agree on direction before normalization.
 
### 10. Dataset choice reaffirmed: staying on CICIoT2023
Checked current alternatives within a 5-6 year window: TII-SSRC-23 (2023), Edge-IIoTset (2022),
and Gotham Dataset 2025 (a newer IoT testbed that captures traffic separately per device — its
real advantage is that it *would* support genuine per-source sequences, unlike CICIoT2023).
Not switching: doing so would discard a completed EDA and a locked, validated model plan, and
Gotham would still require building a PCAP→CSV feature-extraction pipeline ourselves to get
anything beyond raw packets — the same scope problem decision #7 already ruled out. CICIoT2023
also remains the most widely cited IoT IDS benchmark from this period, which matters for the
report's related-work section. Noted as a genuine Future Scope item: "device-separated datasets
such as Gotham 2025 could enable real temporal sequence modeling in future work."
 
### 11. `ml/sequence_builder.py` removed from the repo
Direct consequence of #6/#8 — no model in the current design requires windowed/sequential input,
so the module has no purpose. All three models consume the same flat, scaled feature matrix.
