#!/usr/bin/env bash
set -euo pipefail
# Download missing files locally, then run this script for resumable transfer.
# The destination is never accepted until both sides have matching SHA256.
SRC=${1:?local file required}
DST=${2:?remote relative path required}
REMOTE='ai@v100-ssh.apexolab.com'
ssh -o 'ProxyCommand=cloudflared access ssh --hostname %h' "$REMOTE" "mkdir -p /data/mini-k3/data/raw/$(dirname \"$DST\")"
rsync -avP -e "ssh -o ProxyCommand='cloudflared access ssh --hostname %h'" "$SRC" "$REMOTE:/data/mini-k3/data/raw/$DST"
LOCAL_SUM=$(shasum -a 256 "$SRC" | awk '{print $1}')
REMOTE_SUM=$(ssh -o 'ProxyCommand=cloudflared access ssh --hostname %h' "$REMOTE" "sha256sum /data/mini-k3/data/raw/$DST" | awk '{print $1}')
test "$LOCAL_SUM" = "$REMOTE_SUM"
echo "verified $DST $LOCAL_SUM"
