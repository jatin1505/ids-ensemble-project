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
