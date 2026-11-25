#!/bin/bash
set -euo pipefail

OUTPUT="${1:-results/environment.txt}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$(dirname "$OUTPUT")"
{
  echo "timestamp=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "working_directory=$(pwd)"
  echo
  echo "[cpu]"
  lscpu 2>&1 || true
  echo
  echo "[memory]"
  free -h 2>&1 || true
  echo
  echo "[python]"
  "$PYTHON" --version 2>&1 || true
  echo
  echo "[cuda]"
  nvidia-smi 2>&1 || true
  echo
  echo "[packages]"
  "$PYTHON" -m pip freeze 2>&1 || true
} > "$OUTPUT"
