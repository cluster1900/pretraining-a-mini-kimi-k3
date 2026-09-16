#!/usr/bin/env bash
set -euo pipefail
BASE=/data/mini-k3/data
LOG=/data/mini-k3/logs/pipeline-watch.log
SOURCES=(fineweb-edu culturax cosmopedia finemath open-web-math dolma-body code-python openassistant openhermes openr1 ultrafeedback)
while true; do
  ready=0
  for s in "${SOURCES[@]}"; do
    if [[ -f "$BASE/cleaned/$s.jsonl.report.json" ]]; then
      if [[ ! -f "$BASE/deduped/$s.jsonl.report.json" ]] && ! pgrep -f "deduplicate.py --input $BASE/cleaned/$s.jsonl" >/dev/null; then
        nohup /data/mini-k3/venv/bin/python -u /data/mini-k3/project/train/data/deduplicate.py --input "$BASE/cleaned/$s.jsonl" --output "$BASE/deduped/$s.jsonl" >>"$LOG" 2>&1 &
      fi
    fi
    [[ -f "$BASE/deduped/$s.jsonl.report.json" ]] && ready=$((ready+1))
  done
  echo "$(date -Is) dedup_ready=$ready/${#SOURCES[@]}" >>"$LOG"
  [[ "$ready" -eq "${#SOURCES[@]}" ]] && exit 0
  sleep 60
done
