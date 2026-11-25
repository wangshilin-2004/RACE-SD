#!/bin/bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"

# Preserve explicit environment overrides when loading the defaults.
names=(PYTHON DRAFT_MODEL TARGET_7B DEVICE_DRAFT DEVICE_TARGET SEED MAX_LEN)
saved_names=()
saved_values=()
for name in "${names[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    saved_names+=("$name")
    saved_values+=("${!name}")
  fi
done
if [[ -f "$CODE_DIR/experiment.env" ]]; then
  source "$CODE_DIR/experiment.env"
fi
for ((i=0; i<${#saved_names[@]}; i++)); do
  printf -v "${saved_names[$i]}" '%s' "${saved_values[$i]}"
done
STAGE="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${PYTHON:-python3}" "$CODE_DIR/supervisor_audits.py" "$STAGE" \
  --output-dir "${AUDIT_RESULTS_DIR:-$CODE_DIR/results_icc2027_supervisor_r2}" \
  --prompt-file "${PROMPTS:-$CODE_DIR/prompts_300.jsonl}" \
  --draft-model "${DRAFT_MODEL:-models/vicuna-68m}" \
  --target-model "${TARGET_7B:-models/vicuna-7b-v1.5}" \
  --device-draft "${DEVICE_DRAFT:-cpu}" --device-target "${DEVICE_TARGET:-cuda:0}" \
  --seed "${SEED:-321}" --max-len "${MAX_LEN:-80}" \
  --sensitivity-per-workload "${AUDIT_SENSITIVITY_PER_WORKLOAD:-30}" \
  --confirmation-per-workload "${AUDIT_CONFIRMATION_PER_WORKLOAD:-10}" \
  --precision-per-workload "${PRECISION_PER_WORKLOAD:-10}" \
  --precision-repeats "${PRECISION_REPEATS:-2}" \
  --timing-per-workload "${TIMING_PER_WORKLOAD:-10}" \
  --seconds-good "${EST_SECONDS_PER_RUN:-10}" \
  --seconds-poor "${EST_SECONDS_POOR_RUN:-15}" \
  --seconds-int8 "${EST_SECONDS_INT8_RUN:-15}" "$@"
