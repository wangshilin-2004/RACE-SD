#!/bin/bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"
if [[ -f "$CODE_DIR/experiment.env" ]]; then
  # shellcheck disable=SC1091
  source "$CODE_DIR/experiment.env"
fi

STAGE="${1:-sanity}"
PYTHON="${PYTHON:-python3}"
PROMPTS="${PROMPTS:-$CODE_DIR/prompts_300.jsonl}"
OUT="${PIPESD_RESULTS_DIR:-$CODE_DIR/results_icc2027_pipesd_r1}"
DRAFT_MODEL="${DRAFT_MODEL:-models/vicuna-68m}"
TARGET_7B="${TARGET_7B:-models/vicuna-7b-v1.5}"
DEVICE_DRAFT="${DEVICE_DRAFT:-cpu}"
DEVICE_TARGET="${DEVICE_TARGET:-cuda:0}"
PIPE_SD_SINGLE="${PIPE_SD_SINGLE:-0.94}"
PIPE_SD_MULTI="${PIPE_SD_MULTI:-0.90}"
PIPE_SD_MAX_GAMMA="${PIPE_SD_MAX_GAMMA:-8}"
LINK_TRACE_PATH="${LINK_TRACE_PATH:-$CODE_DIR/true_wifi_move_trace.json}"

mkdir -p "$OUT"
/bin/bash "$CODE_DIR/capture_environment.sh" "$OUT/environment.txt"
exec > >(tee -a "$OUT/pipesd_${STAGE}.log") 2>&1

common=(
  --prompt-file "$PROMPTS"
  --draft-model "$DRAFT_MODEL"
  --target-model "$TARGET_7B"
  --device-draft "$DEVICE_DRAFT"
  --device-target "$DEVICE_TARGET"
  --seed 321
  --max-len 80
  --model-dtype auto
  --temperature 0.7
  --sampling-top-k 0
  --sampling-top-p 0.0
  --methods pipesd
  --pipesd-min-gamma 1
  --pipesd-max-gamma "$PIPE_SD_MAX_GAMMA"
  --pipesd-packet-dtype float16
)

run_batch() {
  "$PYTHON" "$CODE_DIR/batch_experiments.py" "${common[@]}" "$@"
}

finalize() {
  local path="$1"
  local prompts="$2"
  "$PYTHON" "$CODE_DIR/validate_results.py" "$path" --expected-prompts "$prompts"
  "$PYTHON" "$CODE_DIR/summarize_results.py" "$path" \
    --output "${path%.csv}_summary.csv"
}

run_sanity() {
  "$PYTHON" "$CODE_DIR/test_pipesd_baseline.py"
  run_batch --per-workload-limit 1 --max-len 16 \
    --pipesd-single-threshold "$PIPE_SD_SINGLE" \
    --pipesd-multi-threshold "$PIPE_SD_MULTI" \
    --run-tag pipesd_sanity --output "$OUT/pipesd_sanity.csv"
  finalize "$OUT/pipesd_sanity.csv" 3
}

run_pilot() {
  local single multi
  while read -r single multi; do
    run_batch --per-workload-limit 10 \
      --pipesd-single-threshold "$single" \
      --pipesd-multi-threshold "$multi" \
      --run-tag "pipesd_pilot_s${single}_m${multi}" \
      --output "$OUT/pipesd_pilot.csv"
  done <<'EOF'
0.90 0.85
0.94 0.90
0.97 0.90
0.94 0.95
EOF
  finalize "$OUT/pipesd_pilot.csv" 30
  "$PYTHON" "$CODE_DIR/select_pipesd_thresholds.py" "$OUT/pipesd_pilot.csv"
}

run_main() {
  run_batch --pipesd-single-threshold "$PIPE_SD_SINGLE" \
    --pipesd-multi-threshold "$PIPE_SD_MULTI" \
    --run-tag "pipesd_main_s${PIPE_SD_SINGLE}_m${PIPE_SD_MULTI}" \
    --output "$OUT/pipesd_main.csv"
  finalize "$OUT/pipesd_main.csv" 300
}

run_network() {
  local name bandwidth rtt
  while read -r name bandwidth rtt; do
    run_batch --bandwidth "$bandwidth" --rtt-ms "$rtt" \
      --pipesd-single-threshold "$PIPE_SD_SINGLE" \
      --pipesd-multi-threshold "$PIPE_SD_MULTI" \
      --run-tag "pipesd_network_${name}" \
      --output "$OUT/pipesd_network.csv"
  done <<'EOF'
good 100 20
moderate 25 50
poor 5 100
EOF
  finalize "$OUT/pipesd_network.csv" 300
}

run_dynamic_trace() {
  run_batch --link-trace-path "$LINK_TRACE_PATH" --link-trace-speedup 10 \
    --pipesd-single-threshold "$PIPE_SD_SINGLE" \
    --pipesd-multi-threshold "$PIPE_SD_MULTI" \
    --run-tag pipesd_dynamic_trace \
    --output "$OUT/pipesd_dynamic_trace.csv"
  finalize "$OUT/pipesd_dynamic_trace.csv" 300
}

case "$STAGE" in
  sanity) run_sanity ;;
  pilot) run_pilot ;;
  main) run_main ;;
  network) run_network ;;
  dynamic_trace) run_dynamic_trace ;;
  *) echo "Usage: $0 {sanity|pilot|main|network|dynamic_trace}" >&2; exit 2 ;;
esac
