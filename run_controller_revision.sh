#!/bin/bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"
if [[ -f "$CODE_DIR/experiment.env" ]]; then
  source "$CODE_DIR/experiment.env"
fi
PYTHON="${PYTHON:-python3}"
STAGE="${1:-sanity}"
OUT="${REVISION_RESULTS_DIR:-$CODE_DIR/results_icc2027_controller_r1}"

case "$STAGE" in
  unit)
    "$PYTHON" "$CODE_DIR/test_controller_revision.py"
    exit 0
    ;;
  sanity|pilot|all) ;;
  *) echo "Usage: /bin/bash run_controller_revision.sh {unit|sanity|pilot|all}"; exit 2 ;;
esac

mkdir -p "$OUT"
export PYTHON
/bin/bash "$CODE_DIR/capture_environment.sh" "$OUT/environment_${STAGE}.txt"
exec > >(tee -a "$OUT/${STAGE}.log") 2>&1

common=(
  --prompt-file "${PROMPTS:-$CODE_DIR/prompts_300.jsonl}"
  --draft-model "${DRAFT_MODEL:-models/vicuna-68m}"
  --target-model "${TARGET_7B:-models/vicuna-7b-v1.5}"
  --device-draft "${DEVICE_DRAFT:-cpu}"
  --device-target "${DEVICE_TARGET:-cuda:0}"
  --seed "${SEED:-321}" --model-dtype "${MODEL_DTYPE:-auto}"
  --temperature "${TEMPERATURE:-0.7}"
  --sampling-top-k "${SAMPLING_TOP_K:-0}" --sampling-top-p "${SAMPLING_TOP_P:-0.0}"
  --async-shadow-launch-mode "${ASYNC_LAUNCH_MODE:-adaptive}"
  --async-shadow-anchor-mode "${ASYNC_ANCHOR_MODE:-sample}"
  --async-shadow-overlap-margin "${ASYNC_OVERLAP_MARGIN:-1.0}"
  --controller-matrix --method-order rotate
)

run_sanity() {
  echo "Sanity: 3 prompts x 5 variants x 2 methods = 30 runs, 16 tokens each."
  "$PYTHON" "$CODE_DIR/test_controller_revision.py"
  "$PYTHON" "$CODE_DIR/test_shadow_pipeline.py"
  "$PYTHON" "$CODE_DIR/batch_experiments.py" "${common[@]}" \
    --per-workload-limit 1 --max-len 16 --bandwidth 100 --rtt-ms 20 \
    --methods dynamic_dssd,async_sd --run-tag revision_sanity --output "$OUT/sanity.csv"
  "$PYTHON" "$CODE_DIR/validate_results.py" "$OUT/sanity.csv" --expected-prompts 3
  "$PYTHON" "$CODE_DIR/check_controller_revision.py" "$OUT/sanity.csv" \
    --expected-prompts 3 --methods dynamic_dssd,async_sd --estimate-pilot
}

run_pilot() {
  "$PYTHON" "$CODE_DIR/check_controller_revision.py" "$OUT/sanity.csv" \
    --expected-prompts 3 --methods dynamic_dssd,async_sd --estimate-pilot
  echo "Pilot: 30 prompts x 5 variants x 2 links = 300 runs, 80 tokens each."
  for link in good poor; do
    bandwidth=100; rtt=20
    if [[ "$link" == poor ]]; then bandwidth=5; rtt=100; fi
    "$PYTHON" "$CODE_DIR/batch_experiments.py" "${common[@]}" \
      --per-workload-limit 10 --max-len 80 --bandwidth "$bandwidth" --rtt-ms "$rtt" \
      --methods async_sd --run-tag "revision_pilot_${link}" --output "$OUT/pilot_${link}.csv"
    "$PYTHON" "$CODE_DIR/validate_results.py" "$OUT/pilot_${link}.csv" --expected-prompts 30
    "$PYTHON" "$CODE_DIR/check_controller_revision.py" "$OUT/pilot_${link}.csv" \
      --expected-prompts 30 --methods async_sd
    "$PYTHON" "$CODE_DIR/summarize_results.py" "$OUT/pilot_${link}.csv" \
      --output "$OUT/pilot_${link}_summary.csv"
  done
}

if [[ "$STAGE" == sanity || "$STAGE" == all ]]; then run_sanity; fi
if [[ "$STAGE" == pilot || "$STAGE" == all ]]; then run_pilot; fi
