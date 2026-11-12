"""
tests/test_feature_extractor.py

Covers live/feature_extractor.py and live/extract_pcap.py with SYNTHETIC
packets built by hand, so every expected number below can be checked on paper.
The expected values follow docs/EXTRACTION_SPEC.md.

Run from the project root:
    python -m pytest tests/test_feature_extractor.py
"""
import csv
import json
import math
import random
from pathlib import Path

import dpkt
import pytest

from live import extract_pcap
from live.feature_extractor import (
    BLOCK_SIZE, COMPUTED, CONSTANT_ZERO, FEATURE_COLUMNS, NEUTRALIZED, WINDOW_SIZE,
    WindowExtractor, extract_pcap as extract_pcap_windows, extract_windows, load_feature_columns,
)

ROOT = Path(__file__).resolve().parent.parent
SAVED_COLUMNS = ROOT / "ml" / "saved_models" / "feature_columns.json"

SRC_IP, DST_IP = b"\xc0\xa8\x4d\x14", b"\xc0\xa8\x4d\x0a"   # 192.168.77.20 -> 192.168.77.10
HEADERS = 14 + 20                                           # Ethernet + IPv4


# ---------------------------------------------------------------- frame builders
def _ethernet(payload, eth_type=dpkt.ethernet.ETH_TYPE_IP) -> bytes:
    eth = dpkt.ethernet.Ethernet(src=b"\x02\x00\x00\x00\x00\x01", dst=b"\x02\x00\x00\x00\x00\x02",
                                 type=eth_type, data=payload)
    return bytes(eth)


def tcp_frame(length=54, sport=40000, dport=443, flags=dpkt.tcp.TH_ACK, ttl=64) -> bytes:
    """A TCP frame of exactly `length` bytes (the minimum is 54: Ethernet + IP + TCP headers)."""
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=b"x" * (length - HEADERS - 20))
    ip = dpkt.ip.IP(src=SRC_IP, dst=DST_IP, p=dpkt.ip.IP_PROTO_TCP, ttl=ttl, data=tcp)
    ip.len = len(ip)
    return _ethernet(ip)


def udp_frame(length=60, sport=40000, dport=53, ttl=64) -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=b"x" * (length - HEADERS - 8))
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(src=SRC_IP, dst=DST_IP, p=dpkt.ip.IP_PROTO_UDP, ttl=ttl, data=udp)
    ip.len = len(ip)
    return _ethernet(ip)


def icmp_frame(length=60, ttl=64) -> bytes:
    echo = dpkt.icmp.ICMP.Echo(id=1, seq=1, data=b"x" * (length - HEADERS - 8))
    icmp = dpkt.icmp.ICMP(type=8, code=0, data=echo)
    ip = dpkt.ip.IP(src=SRC_IP, dst=DST_IP, p=dpkt.ip.IP_PROTO_ICMP, ttl=ttl, data=icmp)
    ip.len = len(ip)
    return _ethernet(ip)


def arp_frame() -> bytes:
    arp = dpkt.arp.ARP(sha=b"\x02\x00\x00\x00\x00\x01", spa=SRC_IP,
                       tha=b"\x00" * 6, tpa=DST_IP, op=dpkt.arp.ARP_OP_REQUEST)
    return _ethernet(arp, dpkt.ethernet.ETH_TYPE_ARP)


def ipv6_frame() -> bytes:
    return _ethernet(b"\x60" + b"\x00" * 39, dpkt.ethernet.ETH_TYPE_IP6)


def stream(frames, start=1000.0, step=0.1):
    return [(start + i * step, f) for i, f in enumerate(frames)]


def windows_of(frames, **kwargs):
    return list(extract_windows(stream(frames), **kwargs))


def single(frames, **kwargs):
    """The one window the frames make; fails if there is not exactly one."""
    windows = windows_of(frames, **kwargs)
    assert len(windows) == 1
    return windows[0].features


# ---------------------------------------------------------------- the column groups
def test_the_three_groups_partition_the_46_columns():
    assert len(FEATURE_COLUMNS) == 46 and len(set(FEATURE_COLUMNS)) == 46
    assert (len(COMPUTED), len(CONSTANT_ZERO), len(NEUTRALIZED)) == (25, 8, 13)
    assert set(COMPUTED) | set(CONSTANT_ZERO) | set(NEUTRALIZED) == set(FEATURE_COLUMNS)
    assert set(COMPUTED).isdisjoint(CONSTANT_ZERO) and set(COMPUTED).isdisjoint(NEUTRALIZED)
    assert set(CONSTANT_ZERO).isdisjoint(NEUTRALIZED)


@pytest.mark.skipif(not SAVED_COLUMNS.exists(), reason="ml/saved_models/feature_columns.json not present")
def test_columns_match_the_columns_the_models_were_trained_on():
    assert load_feature_columns(SAVED_COLUMNS) == json.loads(SAVED_COLUMNS.read_text())


def test_a_mismatching_saved_column_file_is_rejected(tmp_path):
    bad = tmp_path / "feature_columns.json"
    bad.write_text(json.dumps(FEATURE_COLUMNS[::-1]))
    with pytest.raises(ValueError):
        load_feature_columns(bad)


def test_output_is_in_model_order_with_nan_for_neutralized_and_zero_for_constants():
    f = single([tcp_frame()] * WINDOW_SIZE)
    assert list(f.index) == FEATURE_COLUMNS
    assert f[NEUTRALIZED].isna().all()
    assert (f[CONSTANT_ZERO] == 0.0).all()
    assert f[COMPUTED].notna().all()


# ---------------------------------------------------------------- windows
def test_a_window_is_exactly_ten_packets_and_a_partial_window_is_dropped_by_default():
    assert windows_of([tcp_frame()] * 9) == []
    assert len(windows_of([tcp_frame()] * 10)) == 1
    assert len(windows_of([tcp_frame()] * 25)) == 2
    partial = windows_of([tcp_frame()] * 25, emit_partial=True)
    assert [w.n_packets for w in partial] == [10, 10, 5]


def test_window_timestamps_are_those_of_its_first_and_last_packet():
    w = windows_of([tcp_frame()] * 10)[0]
    assert w.ts_start == pytest.approx(1000.0)
    assert w.ts_end == pytest.approx(1000.9)


def test_same_input_gives_the_same_output():
    frames = [tcp_frame(54 + (i * 7) % 90) for i in range(40)]
    a, b = windows_of(frames), windows_of(frames)
    assert len(a) == len(b) == 4
    for x, y in zip(a, b):
        assert x.features.equals(y.features)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        WindowExtractor(window_size=0)
    with pytest.raises(ValueError):
        WindowExtractor(block_size=0)


# ---------------------------------------------------------------- hand-checked values
def test_ten_identical_tcp_packets_give_hand_checked_values():
    f = single([tcp_frame(54, dport=443, ttl=64)] * 10)
    assert f["Tot size"] == 54 and f["Min"] == 54 and f["Max"] == 54 and f["AVG"] == 54
    # running sums over the block are 54, 108, ..., 540: their mean is 54 * 5.5
    assert f["Tot sum"] == pytest.approx(54 * 5.5)
    assert f["Std"] == 0 and f["Radius"] == 0 and f["Variance"] == 0
    assert f["Magnitue"] == pytest.approx(math.sqrt(2 * 54))
    assert f["Protocol Type"] == 6 and f["Duration"] == 64
    for name in ("TCP", "HTTPS", "ack_flag_number", "IPv", "LLC"):
        assert f[name] == 1.0, name
    for name in ("HTTP", "UDP", "ICMP", "ARP", "DNS", "syn_flag_number", "fin_flag_number",
                 "rst_flag_number", "psh_flag_number"):
        assert f[name] == 0.0, name


def test_variance_is_zero_for_the_first_packet_of_a_block_and_one_when_lengths_differ():
    # lengths 54, 64, 54, 64, ...: packet 1 has one length (variance 0), packets 2..10 have two
    f = single([tcp_frame(54 if i % 2 == 0 else 64) for i in range(10)])
    assert f["Variance"] == pytest.approx(0.9)


def test_the_block_restarts_after_every_20th_packet():
    lengths = [54 + i for i in range(30)]                  # strictly increasing
    windows = windows_of([tcp_frame(n) for n in lengths])
    assert [round(w.features["Variance"], 6) for w in windows] == [0.9, 1.0, 0.9]

    # Third window = packets 21..30. Packet 21 starts a new block, so the running sum
    # for packet 21 is its own length, for packet 22 the sum of 21 and 22, and so on.
    block = lengths[20:30]
    running_sums = [sum(block[: i + 1]) for i in range(10)]
    assert windows[2].features["Tot sum"] == pytest.approx(sum(running_sums) / 10)
    # and Min for packets 21..30 is the first length of that new block each time
    assert windows[2].features["Min"] == pytest.approx(block[0])


def test_running_statistics_use_the_population_standard_deviation():
    # Two packets, lengths 54 and 74: mean 64, population std 10 (sample std would be 14.14)
    ext = WindowExtractor(window_size=2)
    ext.add(1.0, tcp_frame(54))
    w = ext.add(2.0, tcp_frame(74))
    # packet 1: std 0; packet 2: std 10 -> window mean 5
    assert w.features["Std"] == pytest.approx(5.0)
    # Radius = sqrt(2) * Std per packet, so also for the window mean
    assert w.features["Radius"] == pytest.approx(math.sqrt(2) * 5.0)


def test_a_mixed_window_gives_hand_checked_fractions():
    frames = ([tcp_frame(dport=443)] * 4 + [udp_frame(dport=53)] * 3
              + [icmp_frame()] * 2 + [arp_frame()])
    f = single(frames)
    expected = {"TCP": 0.4, "UDP": 0.3, "ICMP": 0.2, "ARP": 0.1, "IPv": 0.9, "LLC": 0.9,
                "HTTPS": 0.4, "DNS": 0.3, "ack_flag_number": 0.4}
    for name, value in expected.items():
        assert f[name] == pytest.approx(value), name
    assert f["Protocol Type"] == pytest.approx((4 * 6 + 3 * 17 + 2 * 1 + 0) / 10)   # 7.7
    assert f["Duration"] == pytest.approx(9 * 64 / 10)                              # ARP has TTL 0


def test_ipv_and_llc_are_identical_because_of_a_quirk_of_the_original_code():
    f = single([tcp_frame()] * 6 + [arp_frame()] * 2 + [udp_frame()] * 2)
    assert f["IPv"] == f["LLC"] == pytest.approx(0.8)


# ---------------------------------------------------------------- single-packet rules
def one(frame):
    return single([frame], window_size=1)


@pytest.mark.parametrize("flags, expected", [
    (dpkt.tcp.TH_SYN, {"syn_flag_number": 1, "ack_flag_number": 0}),
    (dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK, {"syn_flag_number": 1, "ack_flag_number": 1}),
    (dpkt.tcp.TH_FIN | dpkt.tcp.TH_ACK, {"fin_flag_number": 1, "ack_flag_number": 1}),
    (dpkt.tcp.TH_RST, {"rst_flag_number": 1, "ack_flag_number": 0}),
    (dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, {"psh_flag_number": 1, "ack_flag_number": 1}),
])
def test_tcp_flags(flags, expected):
    f = one(tcp_frame(flags=flags))
    for name in ("fin_flag_number", "syn_flag_number", "rst_flag_number",
                 "psh_flag_number", "ack_flag_number"):
        assert f[name] == expected.get(name, 0), name


def test_port_rules_for_http_https_and_dns():
    assert (one(tcp_frame(dport=80))["HTTP"], one(tcp_frame(dport=80))["HTTPS"]) == (1, 0)
    assert (one(tcp_frame(sport=443, dport=50000))["HTTPS"]) == 1          # server to client
    assert one(udp_frame(dport=53))["DNS"] == 1
    assert one(udp_frame(sport=53, dport=50000))["DNS"] == 1
    assert one(udp_frame(dport=443))["HTTPS"] == 0                          # HTTPS is a TCP rule
    assert one(tcp_frame(dport=53))["DNS"] == 0                             # DNS is a UDP rule


def test_arp_sets_arp_but_not_the_ip_features():
    f = one(arp_frame())
    assert f["ARP"] == 1 and f["IPv"] == 0 and f["LLC"] == 0
    assert f["Protocol Type"] == 0 and f["Duration"] == 0
    assert f["Tot size"] == 42                                              # Ethernet + ARP


def test_udp_and_icmp_packets():
    u, i = one(udp_frame(60)), one(icmp_frame(60))
    assert (u["UDP"], u["TCP"], u["ICMP"], u["Protocol Type"]) == (1, 0, 0, 17)
    assert (i["ICMP"], i["TCP"], i["UDP"], i["Protocol Type"]) == (1, 0, 0, 1)
    assert u["Tot size"] == 60 and i["Tot size"] == 60


# ---------------------------------------------------------------- skipping
def test_ipv6_and_unreadable_frames_are_skipped_and_do_not_count_toward_a_window():
    ext = WindowExtractor()
    windows = []
    for i in range(10):
        windows.append(ext.add(float(i), tcp_frame()))
        ext.add(float(i) + 0.5, ipv6_frame())
        ext.add(float(i) + 0.6, b"\x01\x02\x03")
    assert sum(w is not None for w in windows) == 1
    assert ext.skipped == 20


# ---------------------------------------------------------------- structural invariants
def random_frames(n, seed):
    rng = random.Random(seed)
    makers = [
        lambda: tcp_frame(rng.randint(54, 1514), dport=rng.choice([80, 443, 8080]),
                          flags=rng.choice([dpkt.tcp.TH_ACK, dpkt.tcp.TH_SYN,
                                            dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK, dpkt.tcp.TH_FIN])),
        lambda: udp_frame(rng.randint(50, 600), dport=rng.choice([53, 5000])),
        lambda: icmp_frame(rng.randint(42, 200)),
        arp_frame,
    ]
    return [rng.choice(makers)() for _ in range(n)]


def test_structural_invariants_hold_on_every_window_of_random_traffic():
    windows = windows_of(random_frames(600, seed=1))
    assert len(windows) == 60
    for w in windows:
        f = w.features
        assert f["Min"] <= f["AVG"] + 1e-9 and f["AVG"] <= f["Max"] + 1e-9
        assert f["Min"] - 1e-9 <= f["Tot size"] <= f["Max"] + 1e-9
        assert f["Std"] <= (f["Max"] - f["Min"]) / 2 + 1e-9          # population std
        assert 0.0 <= f["Variance"] <= 1.0
        assert f["Variance"] * 10 == pytest.approx(round(f["Variance"] * 10))   # multiples of 1/10
        assert f["Radius"] == pytest.approx(math.sqrt(2) * f["Std"])
        assert f["Magnitue"] ** 2 <= 2 * f["AVG"] + 1e-6              # mean of square roots <= root of mean
        assert f["IPv"] == f["LLC"]


# ---------------------------------------------------------------- pcap files
def write_pcap(path, frames, start=1000.0, step=0.1, linktype=dpkt.pcap.DLT_EN10MB):
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f, linktype=linktype)
        for i, frame in enumerate(frames):
            writer.writepkt(frame, ts=start + i * step)


def test_pcap_round_trip_matches_the_in_memory_result(tmp_path):
    frames = random_frames(100, seed=2)
    path = tmp_path / "t.pcap"
    write_pcap(path, frames)
    from_file = list(extract_pcap_windows(path))
    in_memory = windows_of(frames)
    assert len(from_file) == len(in_memory) == 10
    for a, b in zip(from_file, in_memory):
        assert a.features.equals(b.features)
        assert a.ts_start == pytest.approx(b.ts_start, abs=1e-5)


def test_a_capture_cut_off_mid_packet_does_not_crash(tmp_path):
    path = tmp_path / "t.pcap"
    write_pcap(path, [tcp_frame()] * 50)
    cut = tmp_path / "cut.pcap"
    cut.write_bytes(path.read_bytes()[:-7])
    windows = list(extract_pcap_windows(cut))
    assert 1 <= len(windows) <= 5


def test_a_non_ethernet_capture_is_rejected(tmp_path):
    path = tmp_path / "t.pcap"
    write_pcap(path, [tcp_frame()] * 10, linktype=113)   # Linux "cooked" capture, as from -i any
    with pytest.raises(ValueError):
        list(extract_pcap_windows(path))


# ---------------------------------------------------------------- the command-line tool
def test_cli_writes_a_csv_and_labels_windows_with_the_phase_log(tmp_path):
    pcap = tmp_path / "t.pcap"
    write_pcap(pcap, [tcp_frame()] * 100, start=1000.0, step=0.1)   # windows start at 1000, 1001, ...
    phases = tmp_path / "phases.csv"
    phases.write_text(
        "start_epoch,end_epoch,phase,tool,exit_code,params\n"
        '1000.0,1005.0,alpha,locust,0,"first, with a comma"\n'
        '1006.0,1010.0,beta,hping3,124,"second"\n'
    )
    out = tmp_path / "windows.csv"
    assert extract_pcap.main([str(pcap), str(out), "--phases", str(phases)]) == 0

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["ts_start", "ts_end", "n_packets", "phase"] + FEATURE_COLUMNS
    assert len(rows) == 10
    labels = [r["phase"] for r in rows]
    # margin 1.0: alpha holds windows starting at 1001, 1002, 1003; beta those at 1007, 1008
    assert labels.count("alpha") == 3 and labels.count("beta") == 2 and labels.count("unlabeled") == 5
    assert all(r["Magnitue"] != "" for r in rows)
    assert all(r[name] == "" for r in rows for name in NEUTRALIZED)      # NaN written as empty
    assert all(float(r[name]) == 0.0 for r in rows for name in CONSTANT_ZERO)


def test_cli_without_a_phase_log_labels_everything_unlabeled(tmp_path):
    pcap = tmp_path / "t.pcap"
    write_pcap(pcap, [tcp_frame()] * 30)
    out = tmp_path / "windows.csv"
    assert extract_pcap.main([str(pcap), str(out)]) == 0
    with open(out, newline="") as f:
        assert {r["phase"] for r in csv.DictReader(f)} == {"unlabeled"}