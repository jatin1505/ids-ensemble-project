#!/usr/bin/env bash
#
# lab/lab_setup.sh
#
# ONE-TIME setup for the single-machine lab (see docs/LAB_SETUP.md).
# Run as your normal user, WITHOUT sudo (it asks for sudo itself when needed,
# so the Python environment ends up owned by you, not root):
#
#     bash lab/lab_setup.sh
#
# What it installs and prepares:
#   - nginx + a small HTTP/HTTPS site and a self-signed certificate, under
#     /opt/ids-lab. HTTPS matters: in CICIoT2023's benign training data the
#     HTTPS indicator has median 1, so a lab site that only spoke plain HTTP
#     would look less like the training data.
#   - tcpdump, ethtool, iproute2 (capture and network namespaces)
#   - hping3, nmap (flood and scan traffic, LAB USE ONLY, see run_phases.sh)
#   - Locust in ~/lab-venv (normal visitor traffic and the paced HTTP flood)
#
# It also DISABLES the system-wide nginx service so no web server listens on
# your real network interfaces. The lab runs its own nginx, inside a network
# namespace (see lab_up.sh).
#
# Needs an internet connection (apt and pip).

set -euo pipefail

if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    echo "Run this WITHOUT sudo:  bash $0"
    echo "It uses sudo itself, so that the Python environment belongs to your user."
    exit 1
fi
SUDO=""
[ "$EUID" -ne 0 ] && SUDO="sudo"

SERVER_IP="${SERVER_IP:-192.168.77.10}"
LAB_DIR=/opt/ids-lab
VENV="${VENV:-$HOME/lab-venv}"

echo "[1/6] Installing packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y nginx openssl tcpdump ethtool iproute2 hping3 nmap \
    curl python3-venv python3-pip

echo "[2/6] Disabling the system-wide nginx service (the lab runs its own copy)..."
$SUDO systemctl disable --now nginx > /dev/null 2>&1 || true

echo "[3/6] Creating a self-signed certificate for ${SERVER_IP}..."
$SUDO mkdir -p "$LAB_DIR/tls" "$LAB_DIR/www"
$SUDO openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
    -keyout "$LAB_DIR/tls/lab.key" \
    -out "$LAB_DIR/tls/lab.crt" \
    -subj "/CN=${SERVER_IP}" \
    -addext "subjectAltName=IP:${SERVER_IP}" 2> /dev/null
$SUDO chmod 600 "$LAB_DIR/tls/lab.key"

echo "[4/6] Creating the site content in ${LAB_DIR}/www ..."
for page in index about products contact; do
    $SUDO tee "$LAB_DIR/www/${page}.html" > /dev/null <<HTML
<!doctype html>
<html>
<head><title>Lab site: ${page}</title></head>
<body>
<h1>Lab site: ${page}</h1>
<p>Test page for the IDS live-capture lab. Nothing here is real.</p>
<p>
  <a href="/index.html">home</a> |
  <a href="/about.html">about</a> |
  <a href="/products.html">products</a> |
  <a href="/contact.html">contact</a>
</p>
</body>
</html>
HTML
done

# Files of different sizes, so downloads look like a mix of small and large responses.
$SUDO head -c 10240   /dev/urandom | $SUDO tee "$LAB_DIR/www/small.bin"  > /dev/null   # 10 KB
$SUDO head -c 204800  /dev/urandom | $SUDO tee "$LAB_DIR/www/medium.bin" > /dev/null   # 200 KB
$SUDO head -c 2097152 /dev/urandom | $SUDO tee "$LAB_DIR/www/large.bin"  > /dev/null   # 2 MB
$SUDO chmod -R a+rX "$LAB_DIR/www"

echo "[5/6] Writing the lab's nginx configuration..."
# Quoted heredoc: $uri below is nginx syntax, not a shell variable.
$SUDO tee "$LAB_DIR/nginx.conf" > /dev/null <<'NGINX'
user www-data;
worker_processes 2;
pid /run/ids-lab-nginx.pid;
error_log /var/log/ids-lab-nginx-error.log warn;

events {
    worker_connections 4096;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    access_log off;
    sendfile on;
    keepalive_timeout 15;

    server {
        listen 80 default_server;
        listen 443 ssl default_server;
        server_name _;

        ssl_certificate     /opt/ids-lab/tls/lab.crt;
        ssl_certificate_key /opt/ids-lab/tls/lab.key;

        root /opt/ids-lab/www;
        index index.html;

        location / {
            try_files $uri $uri/ =404;
        }
    }
}
NGINX

echo "[6/6] Creating a Python virtual environment in ${VENV} and installing Locust..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet locust

echo
echo "Setup finished. Checks:"
"$VENV/bin/locust" --version
nginx -v 2>&1
hping3 --version 2>&1 | head -n 1 || true
nmap --version | head -n 1
echo
echo "Next: sudo bash lab/lab_up.sh"