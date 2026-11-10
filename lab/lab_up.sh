#!/usr/bin/env bash
#
# lab/lab_up.sh
#
# Brings the lab up (see docs/LAB_SETUP.md). Run after lab_setup.sh:
#
#     sudo bash lab/lab_up.sh
#
# Creates two isolated "virtual machines" using Linux network namespaces,
# joined by a virtual Ethernet cable (a veth pair):
#
#     lab-t  (traffic generator)  192.168.77.20 --- veth-t ===== veth-s --- 192.168.77.10  lab-s (web server)
#
# Each namespace has its own network stack. The only route in either one is
# the 192.168.77.0/24 cable between them, so nothing sent in the lab can reach
# your real network, the college network, or the internet. That is what makes
# the flood and scan phases safe to run on your own laptop.
#
# nginx runs inside lab-s. Packets are captured on veth-s (capture.sh).
#
# Environment variables (all optional):
#     LAB_NET_PREFIX   network prefix, default 192.168.77.
#     OFFLOADS         "off" (default) or "on". With off, the virtual cable does
#                      not merge packets into 64 KB super-packets, so captured
#                      packet lengths look like a normal Ethernet network
#                      (at most 1514 bytes). Leave it off unless you are
#                      deliberately testing the effect of offloading.
#
# Undo everything with:  sudo bash lab/lab_down.sh

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0"
    exit 1
fi

NS_S=lab-s
NS_T=lab-t
IF_S=veth-s
IF_T=veth-t
PREFIX="${LAB_NET_PREFIX:-192.168.77.}"
IP_S="${PREFIX}10"
IP_T="${PREFIX}20"
OFFLOADS="${OFFLOADS:-off}"
NGINX_CONF=/opt/ids-lab/nginx.conf

if [ ! -f "$NGINX_CONF" ]; then
    echo "Missing $NGINX_CONF. Run lab_setup.sh first:  bash lab/lab_setup.sh"
    exit 1
fi

if ip netns list | grep -qE "^(${NS_S}|${NS_T})( |\$)"; then
    echo "The lab already exists. Remove it first:  sudo bash lab/lab_down.sh"
    exit 1
fi

echo "[1/4] Creating the namespaces and the virtual cable..."
ip netns add "$NS_S"
ip netns add "$NS_T"
ip link add "$IF_S" type veth peer name "$IF_T"
ip link set "$IF_S" netns "$NS_S"
ip link set "$IF_T" netns "$NS_T"

echo "[2/4] Assigning addresses (${IP_S} and ${IP_T})..."
ip -n "$NS_S" addr add "${IP_S}/24" dev "$IF_S"
ip -n "$NS_T" addr add "${IP_T}/24" dev "$IF_T"
for pair in "${NS_S}:${IF_S}" "${NS_T}:${IF_T}"; do
    ns="${pair%%:*}"
    ifc="${pair##*:}"
    ip -n "$ns" link set lo up
    ip -n "$ns" link set "$ifc" up
done

echo "[3/4] Offloading: ${OFFLOADS}"
if [ "$OFFLOADS" = "off" ]; then
    for pair in "${NS_S}:${IF_S}" "${NS_T}:${IF_T}"; do
        ns="${pair%%:*}"
        ifc="${pair##*:}"
        ip netns exec "$ns" ethtool -K "$ifc" tso off gso off gro off > /dev/null 2>&1 \
            || echo "    warning: could not change offload settings on ${ifc} (see docs/LAB_SETUP.md, troubleshooting)"
    done
fi

echo "[4/4] Starting nginx inside ${NS_S}..."
if ! nginx -t -c "$NGINX_CONF" > /dev/null 2>&1; then
    echo "nginx rejected its configuration:"
    nginx -t -c "$NGINX_CONF" || true
    echo "The namespaces are up; fix the problem, then run lab_down.sh and lab_up.sh again."
    exit 1
fi
ip netns exec "$NS_S" nginx -c "$NGINX_CONF"

# Wait until the server answers from the other namespace.
code="000"
for _ in $(seq 1 20); do
    code="$(ip netns exec "$NS_T" curl -ks -o /dev/null -w '%{http_code}' --max-time 2 "https://${IP_S}/" || true)"
    [ "$code" = "200" ] && break
    sleep 0.5
done

echo
if [ "$code" = "200" ]; then
    echo "Lab is up. HTTPS from ${NS_T} to ${IP_S} returned 200."
else
    echo "Lab namespaces are up, but the web server did not answer (HTTP code: ${code})."
    echo "Look at /var/log/ids-lab-nginx-error.log"
    exit 1
fi
echo
echo "Next, in two terminals:"
echo "  1)  sudo bash lab/capture.sh session1"
echo "  2)  sudo bash lab/run_phases.sh        (add QUICK=1 for a 2-minute dry run)"