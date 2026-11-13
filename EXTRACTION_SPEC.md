# Extraction Specification: from Packets to CICIoT2023 Feature Rows

**Status as of:** 2026-11-12
**Implements:** `live/feature_extractor.py` (library), `live/extract_pcap.py` (command-line tool)
**Checked by:** `tests/test_feature_extractor.py` (hand-checked values on synthetic packets), `ml/check_live_ranges.py` (live range check)
**Replaces:** the "Supplementary material" claims in `docs/FEATURE_DEFINITIONS.md` and `docs/LIVE_CAPTURE_HANDOFF.md`, which assumed that folder contained the extraction code. It does not (see §1).

**Evidence tags used below**
- `[code]`: the authors' `pcap2csv` code as published (a later, simplified version).
- `[comment]`: the authors' original per-packet logic, which survives as commented-out code inside that script.
- `[notebook]`: the authors' `data_summary` notebook, which shows the original window averaging.
- `[paper]`: Neto et al., *Sensors* 2023, 23(13), 5941 (section 4, Tables 4 and 5).
- `[data]`: observed in the dataset rows by our own scripts.

---

## 1. Sources, and how far to trust each

**The "Supplementary material" folder** (`tools.zip`) holds four short tool descriptions (DPKT, PySpark, tcpdump, mergecap). No code.

**`pcap2csv.zip`** holds the authors' feature-extraction code: `Feature_extraction.py` (the main script), `Dynamic_features.py` (the Magnitude, Radius, covariance, variance-ratio and weight functions), `Layered_features.py` (protocol indicators), `Connectivity_features.py`, `Communication_features.py`, `Supporting_functions.py`, `Generating_dataset.py`, and a data-summary notebook.

**It is not the code that produced the released CSVs.** The published script writes a different set of columns (no `Srate`, `Drate`, `urg_count`, `Magnitue`, `Radius`, `Covariance`, `Weight` or `flow_duration`; the TTL column is called `Time_To_Live`; it adds `IGMP`) and aggregates differently (`Protocol Type` is the mode of a window, `Variance` is the variance of the window's lengths, `Number` is always 10). The released CSVs behave differently on the points that can be checked:

| Check on the released data | Published script would give | Released CSV shows |
|---|---|---|
| `Protocol Type` | a real protocol number (the mode) | benign median 6.5, which is no protocol: it is a mean of 6s and 17s |
| `Variance` | variance of packet lengths (hundreds or thousands) | always in [0, 1], in steps of 1/10 |
| `Number` | exactly 10 | 5.5, 13.5, 9.5, never 10 |

The **older logic survives as commented-out code** in `Feature_extraction.py` `[comment]`, and the notebook shows the original aggregation: every column averaged with `.mean()` over consecutive rows of 10 `[notebook]`. That older logic explains every one of those checks. **We implement the older logic**, because the models were trained on the CSVs.

---

## 2. What the extractor does

**Packets.** Every IPv4 or ARP frame, in capture order. Other frames (IPv6 and so on) are skipped. `[code]`

**Per-packet values** (each then averaged over a window):

| Feature | Rule | Source |
|---|---|---|
| frame length (`Tot size`) | the full Ethernet frame length, without the checksum | `[code]` `[data]` (smallest frames 42 and 54 bytes match ARP and bare TCP frames) |
| `Protocol Type` | IP protocol number (TCP 6, UDP 17, ICMP 1); 0 for ARP | `[code]` |
| `Duration` | the IP **TTL** (not a duration); 0 for ARP | `[code]` `[paper]` |
| `fin/syn/rst/psh/ack_flag_number` | 1 if that TCP flag is set in this packet, else 0 (non-TCP packets 0) | `[code]` |
| `HTTP`, `HTTPS` | TCP and port 80 (or 443) on either side | `[code]` |
| `DNS` | UDP and port 53 on either side | `[code]` |
| `TCP`, `UDP`, `ICMP`, `ARP` | transport or protocol indicators | `[code]` |
| `IPv`, `LLC` | both 1 for every IP packet. The original LLC test is always true (a bug), which is why the two have identical statistics. | `[code]` `[data]` (paper Table 5) |

**Block statistics.** For packet *i* of the current **block of up to 20 consecutive packets**, over the frame lengths of the packets of the block seen so far, including this one: `Tot sum` (sum), `Min`, `Max`, `AVG` (mean), `Std` (**population** standard deviation). The block restarts after its 20th packet. These are **not per flow**; flows play no part. `[comment]` `[data]`

**Group D features**, from the same block. The original code passed the **same list** of frame lengths in as both "incoming" and "outgoing" `[comment]`, so:

| Feature | Value | Source |
|---|---|---|
| `Magnitue` | sqrt(mean + mean) = sqrt(2·AVG) | `[code]` `[comment]` |
| `Radius` | sqrt(variance + variance) = sqrt(2·variance) = √2·Std | `[code]` `[comment]` |
| `Variance` | variance_in / variance_out when the divisor is not 0, else 0. With identical lists this is **1 if the block's lengths are not all equal, else 0**. | `[code]` `[comment]` |
| `Covariance` | covariance of two identical lists = the variance (not computed; neutralized) | `[code]` |
| `Weight` | len(in) × len(out) = n² (not computed; neutralized) | `[code]` |

**Window.** Every **10 consecutive** per-packet rows are averaged column by column into one row. `[notebook]` `[paper]`

**The 46 output columns** are in the order of `ml/saved_models/feature_columns.json`:
- **25 computed:** `Protocol Type`, `Duration`, the five flag numbers, `HTTP`, `HTTPS`, `DNS`, `TCP`, `UDP`, `ARP`, `ICMP`, `IPv`, `LLC`, `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Magnitue`, `Radius`, `Variance`.
- **8 forced to 0** (constant in the benign training data): `Drate`, `ece_flag_number`, `cwr_flag_number`, `Telnet`, `SMTP`, `SSH`, `IRC`, `DHCP`.
- **13 left NaN** and filled later by `backend/model_runtime.py` from `live/live_fill_values.json`: `flow_duration`, `Header_Length`, `Rate`, `Srate`, `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`, `IAT`, `Number`, `Weight`, `Covariance`.

---

## 3. Evidence that the reconstruction is right

Every line is a check against the released data that the older logic predicts and the published script does not.

| Observation in the released CSVs | Explained by |
|---|---|
| `Magnitue² / AVG` never exceeds 2; equals 2 on 98% of flood-dominated rows and 82% of benign rows | per-packet `sqrt(2·AVG)`, then a window mean: the mean of square roots is at most the square root of the mean, with equality when `AVG` is constant over the window |
| `Radius / Std` = 1.414 | per-packet `sqrt(2·variance)` = √2·`Std`; the mean is linear, so the ratio is exact |
| `Variance` takes only multiples of 1/10 on benign rows (100%) and 1/100 on flood rows (100%) | a per-packet 0/1 value averaged over 10 (or 100) packets |
| `Variance` is 0 on every row where `Std` is 0 | variance 0 gives a ratio of 0 |
| about 45% of benign rows have `Variance` 0.9 | a 10-packet window that contains the first packet of a block (variance of one value is 0) |
| `Covariance / Std²` is at least 1 (5th percentile 0.999) | per-packet covariance of identical lists = variance; mean of squares at least square of the mean |
| `Std <= (Max − Min) / 2` on 100% of rows | **population** standard deviation |
| `Min <= Tot size <= Max` on 100% of rows | the block includes the current packet |
| `IPv` and `LLC` have identical statistics | the always-true LLC test |
| `Number` is a constant 9.5 on flood rows | the mean of 0 to 19 over five blocks of 20: this is the main evidence for **block size 20** |
| `Protocol Type` median 6.5 | a mean over a window, not a mode |

---

## 4. What the extractor deliberately does not replicate

- **The 13 neutralized features.** They depend on flow-level counters and timestamps whose original definitions are unclear (`IAT` around 1.67e8, `Header_Length` around 5e5). Detection does not need them (`docs/LIVE_CAPTURE_HANDOFF.md`, ablations).
- **Windows of 100 packets for floods.** The released CSVs use 100 for all DDoS, DoS and Mirai rows. Live traffic cannot know in advance what it is, so **all traffic uses windows of 10**, which is what the benign training rows use (decision D8).
- **Rows for other frame types.** The original script keeps zero-feature rows for some other Ethernet types. We skip them; the lab capture filter only passes IPv4 and ARP anyway.
- **The last partial window.** Dropped by default (`emit_partial=False`), because averaging fewer than 10 packets makes a slightly different kind of row.

---

## 5. Assumptions still open

| Assumption | Why it matters | How it will be checked |
|---|---|---|
| Block size is 20 and the block restarts after the 20th packet | changes `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Magnitue`, `Radius`, `Variance` | §6: try 10, 20, 40 on a dataset PCAP and see which matches the CSV rows |
| ARP frames count toward blocks and windows | affects ARP-heavy traffic only | same test, on an ARP-heavy class |
| Capture order is the order of rows | windows mix packets of different flows, as in the dataset | same test |
| Lab traffic differs from the IoT traffic of the dataset (TTL 64 against a benign median near 99, large frames from file downloads) | explains out-of-range features that are not extractor errors | §6, then `ml/check_live_ranges.py` |

---

## 6. Validation: reproduce a dataset PCAP

The extractor is tested on synthetic packets (every number checked on paper), but that does not show it matches the dataset. The decisive test uses the dataset's own traffic:

1. Download one small PCAP of a class that uses windows of 10 (for example Recon-PingSweep or Uploading_Attack) from the dataset's PCAP directory.
2. Extract it: `python -m live.extract_pcap <that.pcap> windows.csv`.
3. Compare each of the 25 computed features' distribution (median, 25th and 75th percentile) with the CSV rows of the same label. Repeat with `--block 10` and `--block 40`.
4. **Pass:** for at least 22 of 25 features, the extracted median lies between the CSV's 25th and 75th percentile, and block size 20 matches at least as well as the alternatives.

**Why this matters.** Without it, an out-of-range feature in the live range check could mean "our lab traffic is different" or "our extractor is wrong", and we could not tell which. With it, any remaining difference is traffic, not code.

---

## 7. How to use it

```
# extract a lab capture, labeled with the phase log
python -m live.extract_pcap data/live/session1/session1_*.pcap data/live/session1/windows.csv \
    --phases data/live/session1/phases_*.csv

# compare the normal-traffic windows with the benign training ranges
python ml/check_live_ranges.py data/live/session1/windows.csv --by-phase
```

Speed: about 28,000 packets per second in a test (254,000 packets in 9 seconds).

---

## 8. Limits to state in the report

- The released CSVs were produced by an older version of the authors' code that is no longer published. Our extractor is a **reconstruction** from the surviving commented-out logic, the notebook and checks against the data, validated on a dataset PCAP (§6, once done), not a copy.
- 13 of the 46 features are not reproduced; they are filled with benign medians.
- Windows use 10 packets for all traffic, so flood windows differ from the dataset's 100-packet flood rows.
