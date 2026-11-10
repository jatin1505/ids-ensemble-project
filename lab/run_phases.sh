#!/usr/bin/env bash
#
# lab/run_phases.sh
#
# Runs labeled traffic phases from the traffic-generator side of the lab
# against the lab web server, and writes a phase log with the start and end
# time of every phase. The log is the ground truth for the live evaluation:
# it says which traffic was running when.
#
# Usage:     sudo bash lab/run_phases.sh [server-ip]
# Dry run:   sudo QUICK=1 bash lab/run_phases.sh        (about 2 minutes)
#
# The server address defaults to 192.168.77.10. Start capture.sh FIRST in
# another terminal, and stop it only after this script ends.
#
# SAFETY. The flood and scan phases are attack-style traffic. In the normal
# setup they run inside the lab-t namespace, which has no route anywhere
# except the cable to lab-s. As a second line of defence this script also
# refuses any target outside LAB_NET_PREFIX (default 192.168.77.) and refuses
# addresses ending in .1. Never point it at the college network, a cloud
# server, your router, or anything you do not own.
#
# Environment variables (all optional):
#     QUICK=1          short phases, for a dry run
#     NS               namespace to run in. Default lab-t. Set NS= (empty) to run
#                      in the normal namespace instead, for example inside a real VM.
#     LAB_NET_PREFIX   allowed target prefix, default 192.168.77.
#     VENV             Python environment with Locust, default ~/lab-venv
#     OUTDIR           where logs are written, default ~/lab-captures

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0 [server-ip]   (hping3 and nmap need root)"
    exit 1
fi

LAB_NET_PREFIX="${LAB_NET_PREFIX:-192.168.77.}"
SERVER="${1:-${LAB_NET_PREFIX}10}"
NS="${NS-lab-t}"

if [[ "$SERVER" != "${LAB_NET_PREFIX}"* ]] || [[ "$SERVER" == *.1 ]]; then
    echo "Refusing to run against $SERVER."
    echo "Attack-style traffic may only target the lab server (${LAB_NET_PREFIX}x), never a .1 address,"
    echo "the college network, or anything you do not own."
    echo "If your lab uses a different prefix, set LAB_NET_PREFIX (for example: LAB_NET_PREFIX=192.168.99.)."
    exit 1
fi

# Commands run inside the generator namespace, or directly when NS is empty.
NSX=()
if [ -n "$NS" ]; then
    NSX=(ip netns exec "$NS")
    # Check that the namespace exists AND works. A leftover from before a reboot
    # or crash can still be listed by name while being broken.
    if ! "${NSX[@]}" true 2> /dev/null; then
        echo "Namespace ${NS} is missing or broken. Rebuild the lab:"
        echo "  sudo bash lab/lab_down.sh && sudo bash lab/lab_up.sh"
        exit 1
    fi
fi

# ---- Durations -------------------------------------------------------------
NORMAL_LIGHT_SECS=300   # light normal traffic (small, so it can run long)
NORMAL_BUSY_SECS=120    # busy normal traffic (many large downloads: big capture, so shorter)
ATTACK_SECS=60          # each flood or scan phase
GAP=15                  # idle seconds between phases
if [ "${QUICK:-0}" = "1" ]; then
    NORMAL_LIGHT_SECS=20
    NORMAL_BUSY_SECS=20
    ATTACK_SECS=10
    GAP=5
fi

# ---- Paths -------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-root}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
VENV="${VENV:-$REAL_HOME/lab-venv}"
OUTDIR="${OUTDIR:-$REAL_HOME/lab-captures}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$OUTDIR/phases_${STAMP}"
PHASE_LOG="$OUTDIR/phases_${STAMP}.csv"

# ---- Pre-flight checks ---------------------------------------------------------
for tool in hping3 nmap timeout; do
    command -v "$tool" > /dev/null || { echo "Missing tool: $tool (run lab_setup.sh first)"; exit 1; }
done
[ -x "$VENV/bin/locust" ] || { echo "Locust not found in $VENV (run lab_setup.sh first)"; exit 1; }

mkdir -p "$LOGDIR"
echo "start_epoch,end_epoch,phase,tool,exit_code,params" > "$PHASE_LOG"
FAILED=0

# Runs one phase: records its start time, end time and exit code, saves the
# tool's output to its own log file, then waits GAP seconds. Exit code 0 is a
# normal finish and 124 means "stopped by timeout", which is how the flood
# phases normally end. Anything else means the phase probably did not run.
# Usage: run_phase <phase-name> <tool-name> "<params text>" <command...>
run_phase() {
    local name="$1" tool="$2" params="$3"
    shift 3
    echo ">>> ${name}: ${params}"
    local start end
    local code=0
    start="$(date +%s.%N)"
    "$@" > "$LOGDIR/${name}.log" 2>&1 || code=$?
    end="$(date +%s.%N)"
    echo "${start},${end},${name},${tool},${code},\"${params}\"" >> "$PHASE_LOG"
    if [ "$code" -ne 0 ] && [ "$code" -ne 124 ]; then
        echo "    WARNING: ${tool} exited with code ${code}, so this phase probably did not run."
        echo "    See $LOGDIR/${name}.log"
        FAILED=$((FAILED + 1))
    fi
    echo "    finished; idle for ${GAP}s"
    sleep "$GAP"
}

echo "Target: ${SERVER}   namespace: ${NS:-<none>}   normal: ${NORMAL_LIGHT_SECS}s + ${NORMAL_BUSY_SECS}s   attack phases: ${ATTACK_SECS}s   gap: ${GAP}s"
echo "Phase log: ${PHASE_LOG}"
echo

# ---- Normal traffic ------------------------------------------------------------
run_phase normal_light locust "10 users, 2 new per second, ${NORMAL_LIGHT_SECS}s, HTTPS" \
    "${NSX[@]}" "$VENV/bin/locust" -f "$HERE/locustfile.py" --headless --exit-code-on-error 0 \
    -u 10 -r 2 -t "${NORMAL_LIGHT_SECS}s" --host "https://${SERVER}" LabVisitor

run_phase normal_busy locust "40 users, 10 new per second, ${NORMAL_BUSY_SECS}s, HTTPS" \
    "${NSX[@]}" "$VENV/bin/locust" -f "$HERE/locustfile.py" --headless --exit-code-on-error 0 \
    -u 40 -r 10 -t "${NORMAL_BUSY_SECS}s" --host "https://${SERVER}" LabVisitor

# ---- Attack-style traffic (lab only, rate-limited) ----------------------------------
# The HTTP flood is paced by Locust (at most 500 requests per second). hping3
# "-i u500" waits 500 microseconds between packets, about 2000 packets per
# second. Both are enough to look like a flood and gentle enough for a laptop.
run_phase http_flood locust "50 users, up to 500 requests per second, new connection each, ${ATTACK_SECS}s, port 80" \
    "${NSX[@]}" "$VENV/bin/locust" -f "$HERE/locustfile.py" --headless --exit-code-on-error 0 \
    -u 50 -r 50 -t "${ATTACK_SECS}s" --host "http://${SERVER}" HttpFlooder

run_phase syn_flood hping3 "SYN to port 80, about 2000 pps, ${ATTACK_SECS}s" \
    "${NSX[@]}" timeout "$ATTACK_SECS" hping3 -S -p 80 -i u500 "$SERVER"

run_phase udp_flood hping3 "UDP 64-byte payload to port 8080, about 2000 pps, ${ATTACK_SECS}s" \
    "${NSX[@]}" timeout "$ATTACK_SECS" hping3 --udp -p 8080 -d 64 -i u500 "$SERVER"

run_phase icmp_flood hping3 "ICMP echo, about 2000 pps, ${ATTACK_SECS}s" \
    "${NSX[@]}" timeout "$ATTACK_SECS" hping3 --icmp -i u500 "$SERVER"

# --max-rate 1000 keeps the scan to about 1000 packets per second, so 10,000
# ports take about 10 seconds instead of a fraction of a second.
run_phase port_scan nmap "SYN scan of ports 1-10000, max 1000 pps, no DNS" \
    "${NSX[@]}" timeout "$ATTACK_SECS" nmap -sS -p 1-10000 --max-rate 1000 -n "$SERVER"

chown -R "$REAL_USER": "$LOGDIR" "$PHASE_LOG"

echo
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: ${FAILED} phase(s) failed. Do NOT use this session: fix the problem and run it again."
else
    echo "All phases finished."
fi
echo "Phase log:        $PHASE_LOG"
echo "Per-phase output: $LOGDIR/"
echo "Now stop capture.sh (Ctrl+C in its terminal)."