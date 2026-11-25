#!/bin/bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"
if [[ -f "$CODE_DIR/experiment.env" ]]; then
  source "$CODE_DIR/experiment.env"
fi
STAGE="${1:-pilot}"
PAIR="${2:-7b}"
case "$STAGE" in
  smoke) PER_WORKLOAD=1; TOKENS=8; RUNS=9; ESTIMATE="2-5 min" ;;
  pilot) PER_WORKLOAD=10; TOKENS=80; RUNS=90; ESTIMATE="15-30 min" ;;
  full) PER_WORKLOAD=100; TOKENS=80; RUNS=900; ESTIMATE="2-4 hours" ;;
  *) echo "Usage: /bin/bash run_residual_sanity.sh {smoke|pilot|full} {7b|13b}"; exit 2 ;;
esac
case "$PAIR" in
  7b) TARGET="${TARGET_7B:-models/vicuna-7b-v1.5}" ;;
  13b) TARGET="${TARGET_13B:-models/vicuna-13b-v1.5}" ;;
  *) echo "Pair must be 7b or 13b"; exit 2 ;;
esac
PYTHON="${PYTHON:-python3}"
# A separate, unique directory protects all historical performance results.
mkdir -p "$CODE_DIR/results_residual_sanity"
OUT="$(mktemp -d "$CODE_DIR/results_residual_sanity/${STAGE}_${PAIR}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
exec > >(tee "$OUT/run.log") 2>&1
echo "Residual diagnostic: $RUNS runs ($PER_WORKLOAD prompts/workload x 3 workloads x 3 methods)."
echo "Rough 7B estimate: $ESTIMATE; allow longer for 13B. Diagnostic timings are not benchmark results."
echo "Output: $OUT"
"$PYTHON" "$CODE_DIR/test_residual_sanity.py"
/bin/bash "$CODE_DIR/capture_environment.sh" "$OUT/environment.txt"
"$PYTHON" "$CODE_DIR/residual_sanity.py" \
  --prompt-file "${PROMPTS:-$CODE_DIR/prompts_300.jsonl}" \
  --draft-model "${DRAFT_MODEL:-models/vicuna-68m}" \
  --target-model "$TARGET" \
  --device-draft "${DEVICE_DRAFT:-cpu}" --device-target "${DEVICE_TARGET:-cuda:0}" \
  --model-dtype "${MODEL_DTYPE:-auto}" --seed "${SEED:-321}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --sampling-top-k "${SAMPLING_TOP_K:-0}" --sampling-top-p "${SAMPLING_TOP_P:-0.0}" \
  --per-workload-limit "$PER_WORKLOAD" --max-len "$TOKENS" \
  --methods vanilla_dssd,optimized_dssd,async_sd \
  --gamma 4 --bandwidth 100 --rtt-ms 20 \
  --reject-topk 64 --mass-threshold 0.995 --reject-packet-dtype float16 \
  --dynamic-gamma-set "${DYNAMIC_GAMMA_SET:-1,2,3,4,6,8}" \
  --dynamic-min-gamma "${DYNAMIC_MIN_GAMMA:-1}" --safe-gamma "${SAFE_GAMMA:-1}" \
  --async-shadow-launch-mode "${ASYNC_LAUNCH_MODE:-adaptive}" \
  --async-shadow-anchor-mode "${ASYNC_ANCHOR_MODE:-sample}" \
  --async-shadow-overlap-margin "${ASYNC_OVERLAP_MARGIN:-1.0}" \
  --run-tag "${STAGE}_68m_${PAIR}" --output "$OUT/diagnostic.csv"
