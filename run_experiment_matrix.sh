#!/bin/bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"

# Preserve an explicit per-run output override instead of letting experiment.env
# redirect a fresh validation run into an older results directory.
RESULTS_DIR_OVERRIDE="${RESULTS_DIR:-}"
if [[ -f "$CODE_DIR/experiment.env" ]]; then
  # shellcheck disable=SC1091
  source "$CODE_DIR/experiment.env"
fi

STAGE="${1:-sanity}"
PYTHON="${PYTHON:-python3}"
PROMPTS="${PROMPTS:-$CODE_DIR/prompts_300.jsonl}"
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-${RESULTS_DIR:-$CODE_DIR/results_icc2027_v15}}"
DRAFT_MODEL="${DRAFT_MODEL:-models/vicuna-68m}"
DRAFT_160M="${DRAFT_160M:-models/vicuna-160m}"
DRAFT_7B="${DRAFT_7B:-models/vicuna-7b-v1.5}"
TARGET_7B="${TARGET_7B:-models/vicuna-7b-v1.5}"
TARGET_13B="${TARGET_13B:-models/vicuna-13b-v1.5}"
DEVICE_DRAFT="${DEVICE_DRAFT:-cpu}"
DEVICE_TARGET="${DEVICE_TARGET:-cuda:0}"
SEED="${SEED:-321}"
MAX_LEN="${MAX_LEN:-80}"
MODEL_DTYPE="${MODEL_DTYPE:-auto}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-0}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.0}"
SENSITIVITY_PER_WORKLOAD="${SENSITIVITY_PER_WORKLOAD:-30}"
NETWORK_PER_WORKLOAD="${NETWORK_PER_WORKLOAD:-100}"
LINK_TRACE_PATH="${LINK_TRACE_PATH:-$CODE_DIR/true_wifi_move_trace.json}"
DYNAMIC_TRACE_SPEEDUP="${DYNAMIC_TRACE_SPEEDUP:-10}"
LOOKAHEAD_THRESHOLD="${LOOKAHEAD_THRESHOLD:-0.4}"
LOOKAHEAD_MIN_GAMMA="${LOOKAHEAD_MIN_GAMMA:-1}"
LOOKAHEAD_MAX_GAMMA="${LOOKAHEAD_MAX_GAMMA:-8}"
TKSLT_K="${TKSLT_K:-320}"
TKSLT_GAMMA_SET="${TKSLT_GAMMA_SET:-1,2,3,4,6,8}"
TKSLT_BETA="${TKSLT_BETA:-0.2}"
ASYNC_LAUNCH_MODE="${ASYNC_LAUNCH_MODE:-adaptive}"
ASYNC_ANCHOR_MODE="${ASYNC_ANCHOR_MODE:-sample}"
ASYNC_OVERLAP_MARGIN="${ASYNC_OVERLAP_MARGIN:-1.0}"
DYNAMIC_GAMMA_SET="${DYNAMIC_GAMMA_SET:-1,2,3,4,6,8}"
DYNAMIC_MIN_GAMMA="${DYNAMIC_MIN_GAMMA:-1}"
SAFE_GAMMA="${SAFE_GAMMA:-1}"
DECISION_PER_WORKLOAD="${DECISION_PER_WORKLOAD:-10}"
TUNED_GAMMA_13B="${TUNED_GAMMA_13B:-1}"

mkdir -p "$RESULTS_DIR"
/bin/bash "$CODE_DIR/capture_environment.sh" "$RESULTS_DIR/environment.txt"
LOG_STAGE="$STAGE"
if [[ "$STAGE" == "dynamic_trace" ]]; then
  LOG_STAGE="dynamic_trace_final"
fi
exec > >(tee -a "$RESULTS_DIR/${LOG_STAGE}.log") 2>&1

common=(
  --prompt-file "$PROMPTS"
  --draft-model "$DRAFT_MODEL"
  --device-draft "$DEVICE_DRAFT"
  --device-target "$DEVICE_TARGET"
  --seed "$SEED"
  --max-len "$MAX_LEN"
  --model-dtype "$MODEL_DTYPE"
  --temperature "$TEMPERATURE"
  --sampling-top-k "$SAMPLING_TOP_K"
  --sampling-top-p "$SAMPLING_TOP_P"
  --lookahead-confidence-threshold "$LOOKAHEAD_THRESHOLD"
  --lookahead-min-gamma "$LOOKAHEAD_MIN_GAMMA"
  --lookahead-max-gamma "$LOOKAHEAD_MAX_GAMMA"
  --tkslt-k "$TKSLT_K"
  --tkslt-gamma-set "$TKSLT_GAMMA_SET"
  --tkslt-beta "$TKSLT_BETA"
  --async-shadow-launch-mode "$ASYNC_LAUNCH_MODE"
  --async-shadow-anchor-mode "$ASYNC_ANCHOR_MODE"
  --async-shadow-overlap-margin "$ASYNC_OVERLAP_MARGIN"
  --dynamic-gamma-set "$DYNAMIC_GAMMA_SET"
  --dynamic-min-gamma "$DYNAMIC_MIN_GAMMA"
  --safe-gamma "$SAFE_GAMMA"
)

run_batch() {
  "$PYTHON" "$CODE_DIR/batch_experiments.py" "${common[@]}" "$@"
}

finalize_results() {
  local csv_path="$1"
  local expected_prompts="$2"
  local summary_path="${csv_path%.csv}_summary.csv"
  "$PYTHON" "$CODE_DIR/validate_results.py" "$csv_path" --expected-prompts "$expected_prompts"
  "$PYTHON" "$CODE_DIR/summarize_results.py" "$csv_path" --output "$summary_path"
}

run_sanity() {
  "$PYTHON" "$CODE_DIR/test_shadow_pipeline.py"
  "$PYTHON" "$CODE_DIR/sanity_check.py" \
    --prompt-file "$PROMPTS" \
    --draft-model "$DRAFT_MODEL" \
    --target-model "$TARGET_7B" \
    --device-draft "$DEVICE_DRAFT" \
    --device-target "$DEVICE_TARGET" \
    --temperature "$TEMPERATURE" \
    --sampling-top-k "$SAMPLING_TOP_K" \
    --sampling-top-p "$SAMPLING_TOP_P"
  run_batch --target-model "$TARGET_7B" --prompt-limit 1 --max-len 8 \
    --methods dsd,vanilla_dssd,optimized_dssd,tkslt_as2,confidence_lookahead_dssd,dynamic_dssd,async_sd --run-tag sanity \
    --output "$RESULTS_DIR/sanity.csv"
}

run_main() {
  run_batch --target-model "$TARGET_7B" \
    --methods ar_target,dsd,vanilla_dssd,optimized_dssd,tkslt_as2,confidence_lookahead_dssd,dynamic_dssd,async_sd \
    --run-tag main_68m_7b_default_g4 --output "$RESULTS_DIR/main_68m_7b.csv"
  run_batch --target-model "$TARGET_7B" --methods optimized_dssd --gamma 1 \
    --run-tag main_68m_7b_tuned_fixed_g1 --output "$RESULTS_DIR/main_68m_7b.csv"
}

run_pilot() {
  run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len 32 \
    --methods ar_target,dsd,vanilla_dssd,optimized_dssd,tkslt_as2,confidence_lookahead_dssd,dynamic_dssd,async_sd \
    --run-tag pilot_68m_7b --output "$RESULTS_DIR/pilot_68m_7b.csv"
}

run_overlap_pilot() {
  run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len 32 \
    --device-draft cpu --device-target "$DEVICE_TARGET" \
    --methods dynamic_dssd,async_sd \
    --run-tag overlap_pilot_cpu_gpu --output "$RESULTS_DIR/overlap_pilot_cpu_gpu.csv"
}

run_overlap_pilot80() {
  run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len "$MAX_LEN" \
    --device-draft cpu --device-target "$DEVICE_TARGET" \
    --methods dynamic_dssd,async_sd \
    --run-tag overlap_pilot80_cpu_gpu --output "$RESULTS_DIR/overlap_pilot80_cpu_gpu.csv"
}

run_contention_audit() {
  "$PYTHON" "$CODE_DIR/audit_compute_contention.py" \
    --prompt-file "$PROMPTS" \
    --draft-model "$DRAFT_MODEL" \
    --target-model "$TARGET_7B" \
    --device-draft "$DEVICE_DRAFT" \
    --device-target "$DEVICE_TARGET" \
    --model-dtype "$MODEL_DTYPE" \
    --per-workload-limit 2 \
    --repeats 5 \
    --shadow-gamma 1 \
    --output "$RESULTS_DIR/compute_contention_audit.csv"
}

run_shadow_policy_pilot80() {
  run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len "$MAX_LEN" \
    --device-draft cpu --device-target "$DEVICE_TARGET" \
    --methods dynamic_dssd,async_sd --async-shadow-launch-mode adaptive \
    --run-tag shadow_policy_adaptive --output "$RESULTS_DIR/shadow_policy_pilot80.csv"
  run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len "$MAX_LEN" \
    --device-draft cpu --device-target "$DEVICE_TARGET" \
    --methods async_sd --async-shadow-launch-mode always \
    --run-tag shadow_policy_always --output "$RESULTS_DIR/shadow_policy_pilot80.csv"
}

run_shadow_network_pilot() {
  while read -r name bandwidth rtt_ms; do
    run_batch --target-model "$TARGET_7B" --per-workload-limit 3 --max-len "$MAX_LEN" \
      --device-draft cpu --device-target "$DEVICE_TARGET" \
      --methods dynamic_dssd,async_sd --method-order rotate \
      --bandwidth "$bandwidth" --rtt-ms "$rtt_ms" \
      --run-tag "shadow_network_${name}" \
      --output "$RESULTS_DIR/shadow_network_pilot.csv"
  done <<'EOF'
good 100 20
moderate 25 50
poor 5 100
EOF
}

run_shadow_anchor_pilot() {
  local output="$RESULTS_DIR/shadow_anchor_pilot.csv"
  local shared=(
    --target-model "$TARGET_7B"
    --per-workload-limit "$DECISION_PER_WORKLOAD"
    --max-len "$MAX_LEN"
    --device-draft cpu
    --device-target "$DEVICE_TARGET"
    --method-order rotate
  )

  run_batch "${shared[@]}" --methods dynamic_dssd \
    --run-tag shadow_anchor_dynamic --output "$output"
  run_batch "${shared[@]}" --methods async_sd \
    --async-shadow-launch-mode adaptive --async-shadow-anchor-mode sample \
    --run-tag shadow_anchor_sample --output "$output"
  run_batch "${shared[@]}" --methods async_sd \
    --async-shadow-launch-mode adaptive --async-shadow-anchor-mode map \
    --run-tag shadow_anchor_map --output "$output"
}

run_shadow_final_pilot() {
  local output="$RESULTS_DIR/shadow_final_pilot.csv"
  local shared=(
    --target-model "$TARGET_7B"
    --per-workload-limit "$DECISION_PER_WORKLOAD"
    --max-len "$MAX_LEN"
    --device-draft cpu
    --device-target "$DEVICE_TARGET"
  )

  run_batch "${shared[@]}" --methods dynamic_dssd,async_sd \
    --method-order rotate --async-shadow-gamma-cap 1 \
    --async-shadow-launch-mode adaptive --async-shadow-anchor-mode sample \
    --run-tag shadow_uplink_overlap --output "$output"
}

run_shadow_model_pair_pilot() {
  local shared=(
    --target-model "$TARGET_7B"
    --per-workload-limit "$DECISION_PER_WORKLOAD"
    --max-len "$MAX_LEN"
    --device-draft cpu
    --device-target "$DEVICE_TARGET"
    --methods dynamic_dssd,async_sd
    --method-order rotate
    --async-shadow-launch-mode adaptive
    --async-shadow-anchor-mode sample
  )

  run_batch --draft-model "$DRAFT_MODEL" "${shared[@]}" \
    --run-tag shadow_pair_68m_7b \
    --output "$RESULTS_DIR/shadow_pair_68m_7b.csv"
  run_batch --draft-model "$DRAFT_160M" "${shared[@]}" \
    --run-tag shadow_pair_160m_7b \
    --output "$RESULTS_DIR/shadow_pair_160m_7b.csv"
}

run_shadow_cap_pilot() {
  local cap
  for cap in 1 2 8; do
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
      --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
      --device-draft cpu --device-target "$DEVICE_TARGET" \
      --methods dynamic_dssd,async_sd --method-order rotate \
      --async-shadow-launch-mode adaptive --async-shadow-anchor-mode sample \
      --async-shadow-gamma-cap "$cap" \
      --run-tag "shadow_cap_${cap}" \
      --output "$RESULTS_DIR/shadow_cap_pilot.csv"
  done
}

run_model_pair_audit() {
  "$PYTHON" "$CODE_DIR/audit_model_pair.py" "${common[@]}" \
    --pair-tag 68m_7b --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
    --device-draft "$DEVICE_TARGET" --device-target "$DEVICE_TARGET" \
    --per-workload-limit 3 --steps 8 \
    --output "$RESULTS_DIR/model_pair_68m_7b.csv"
  "$PYTHON" "$CODE_DIR/audit_model_pair.py" "${common[@]}" \
    --pair-tag 68m_13b --draft-model "$DRAFT_MODEL" --target-model "$TARGET_13B" \
    --device-draft "$DEVICE_TARGET" --device-target "$DEVICE_TARGET" \
    --per-workload-limit 3 --steps 8 \
    --output "$RESULTS_DIR/model_pair_68m_13b.csv"
  "$PYTHON" "$CODE_DIR/audit_model_pair.py" "${common[@]}" \
    --pair-tag 7b_13b --draft-model "$DRAFT_7B" --target-model "$TARGET_13B" \
    --device-draft "$DEVICE_TARGET" --device-target "$DEVICE_TARGET" \
    --per-workload-limit 3 --steps 8 \
    --output "$RESULTS_DIR/model_pair_7b_13b.csv"
}

run_model_pair_audit_160m() {
  "$PYTHON" "$CODE_DIR/audit_model_pair.py" "${common[@]}" \
    --pair-tag 160m_7b --draft-model "$DRAFT_160M" --target-model "$TARGET_7B" \
    --device-draft "$DEVICE_TARGET" --device-target "$DEVICE_TARGET" \
    --per-workload-limit 3 --steps 8 \
    --output "$RESULTS_DIR/model_pair_160m_7b.csv"
  "$PYTHON" "$CODE_DIR/audit_model_pair.py" "${common[@]}" \
    --pair-tag 160m_13b --draft-model "$DRAFT_160M" --target-model "$TARGET_13B" \
    --device-draft "$DEVICE_TARGET" --device-target "$DEVICE_TARGET" \
    --per-workload-limit 3 --steps 8 \
    --output "$RESULTS_DIR/model_pair_160m_13b.csv"
}

run_pair_pilot_160m() {
  run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_7B" \
    --per-workload-limit 3 --max-len 32 \
    --methods ar_target,dsd,vanilla_dssd,optimized_dssd,tkslt_as2,confidence_lookahead_dssd,dynamic_dssd,async_sd \
    --run-tag pilot_160m_7b --output "$RESULTS_DIR/pilot_160m_7b.csv"
  run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_13B" \
    --per-workload-limit 3 --max-len 32 \
    --methods ar_target,dsd,vanilla_dssd,optimized_dssd,tkslt_as2,confidence_lookahead_dssd,dynamic_dssd,async_sd \
    --run-tag pilot_160m_13b --output "$RESULTS_DIR/pilot_160m_13b.csv"
}

run_low_gamma_pilot() {
  local methods="vanilla_dssd,optimized_dssd,dynamic_dssd,async_sd"
  local gamma_args=(
    --dynamic-gamma-set "1,2,3,4,6,8"
    --dynamic-min-gamma 1
    --safe-gamma 1
  )

  run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
    --per-workload-limit 3 --max-len 32 --methods "$methods" \
    "${gamma_args[@]}" --run-tag low_gamma_68m_7b \
    --output "$RESULTS_DIR/low_gamma_68m_7b.csv"
  run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_13B" \
    --per-workload-limit 3 --max-len 32 --methods "$methods" \
    "${gamma_args[@]}" --run-tag low_gamma_68m_13b \
    --output "$RESULTS_DIR/low_gamma_68m_13b.csv"
  run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_7B" \
    --per-workload-limit 3 --max-len 32 --methods "$methods" \
    "${gamma_args[@]}" --run-tag low_gamma_160m_7b \
    --output "$RESULTS_DIR/low_gamma_160m_7b.csv"
  run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_13B" \
    --per-workload-limit 3 --max-len 32 --methods "$methods" \
    "${gamma_args[@]}" --run-tag low_gamma_160m_13b \
    --output "$RESULTS_DIR/low_gamma_160m_13b.csv"
}

run_decision_pilot() {
  local output="$RESULTS_DIR/decision_pilot_68m_7b.csv"
  run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
    --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
    --methods vanilla_dssd,dynamic_dssd,async_sd \
    --method-order rotate --async-shadow-gamma-cap 1 \
    --run-tag decision_adaptive --output "$output"
  local gamma
  for gamma in 1 2 3 4 6 8; do
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
      --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
      --methods optimized_dssd --gamma "$gamma" \
      --run-tag "decision_fixed_g${gamma}" --output "$output"
  done
}

run_decision_pilot_13b() {
  local output="$RESULTS_DIR/decision_pilot_68m_13b.csv"
  run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_13B" \
    --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
    --methods vanilla_dssd,dynamic_dssd,async_sd \
    --method-order rotate --async-shadow-gamma-cap 1 \
    --run-tag decision_13b_adaptive --output "$output"
  local gamma
  for gamma in 1 2 3 4 6 8; do
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_13B" \
      --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
      --methods optimized_dssd --gamma "$gamma" \
      --run-tag "decision_13b_fixed_g${gamma}" --output "$output"
  done
}

run_controller_fix_pilot() {
  run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
    --per-workload-limit "$DECISION_PER_WORKLOAD" --max-len "$MAX_LEN" \
    --methods optimized_dssd,dynamic_dssd,async_sd --gamma 1 \
    --method-order rotate --async-shadow-gamma-cap 1 \
    --run-tag controller_fix_validation \
    --output "$RESULTS_DIR/controller_fix_pilot.csv"
}

run_controller_network_pilot() {
  local name bandwidth rtt_ms gamma
  while read -r name bandwidth rtt_ms; do
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
      --per-workload-limit 3 --max-len "$MAX_LEN" \
      --bandwidth "$bandwidth" --rtt-ms "$rtt_ms" \
      --methods dynamic_dssd,async_sd --method-order rotate \
      --async-shadow-gamma-cap 1 \
      --run-tag "controller_network_${name}_adaptive" \
      --output "$RESULTS_DIR/controller_network_pilot.csv"
    for gamma in 1 2 3 4 6 8; do
      run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
        --per-workload-limit 3 --max-len "$MAX_LEN" \
        --bandwidth "$bandwidth" --rtt-ms "$rtt_ms" \
        --methods optimized_dssd --gamma "$gamma" \
        --run-tag "controller_network_${name}_fixed_g${gamma}" \
        --output "$RESULTS_DIR/controller_network_pilot.csv"
    done
  done <<'EOF'
good 100 20
moderate 25 50
poor 5 100
EOF
}

run_fixed_gamma() {
  for gamma in 1 2 3 4 6 8; do
    run_batch --target-model "$TARGET_7B" --methods optimized_dssd \
      --gamma "$gamma" --run-tag "fixed_gamma_${gamma}" \
      --output "$RESULTS_DIR/fixed_gamma.csv"
  done
}

run_fixed_gamma_pilot() {
  local gamma
  for gamma in 1 2 3 4 6 8; do
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_7B" \
      --per-workload-limit 3 --max-len 32 --methods optimized_dssd \
      --gamma "$gamma" --run-tag "fixed_gamma_pilot_68m_7b_g${gamma}" \
      --output "$RESULTS_DIR/fixed_gamma_pilot_68m_7b.csv"
    run_batch --draft-model "$DRAFT_MODEL" --target-model "$TARGET_13B" \
      --per-workload-limit 3 --max-len 32 --methods optimized_dssd \
      --gamma "$gamma" --run-tag "fixed_gamma_pilot_68m_13b_g${gamma}" \
      --output "$RESULTS_DIR/fixed_gamma_pilot_68m_13b.csv"
    run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_7B" \
      --per-workload-limit 3 --max-len 32 --methods optimized_dssd \
      --gamma "$gamma" --run-tag "fixed_gamma_pilot_160m_7b_g${gamma}" \
      --output "$RESULTS_DIR/fixed_gamma_pilot_160m_7b.csv"
    run_batch --draft-model "$DRAFT_160M" --target-model "$TARGET_13B" \
      --per-workload-limit 3 --max-len 32 --methods optimized_dssd \
      --gamma "$gamma" --run-tag "fixed_gamma_pilot_160m_13b_g${gamma}" \
      --output "$RESULTS_DIR/fixed_gamma_pilot_160m_13b.csv"
  done
}

run_baselines() {
  run_batch --target-model "$TARGET_7B" \
    --methods tkslt_as2,confidence_lookahead_dssd \
    --run-tag external_baselines_68m_7b --output "$RESULTS_DIR/external_baselines.csv"
}

run_generalization() {
  run_batch --target-model "$TARGET_13B" \
    --methods ar_target,dsd,vanilla_dssd,dynamic_dssd,async_sd \
    --run-tag main_68m_13b --output "$RESULTS_DIR/generalization_13b.csv"
  run_batch --target-model "$TARGET_13B" --methods optimized_dssd \
    --gamma "$TUNED_GAMMA_13B" \
    --run-tag "main_68m_13b_tuned_fixed_g${TUNED_GAMMA_13B}" \
    --output "$RESULTS_DIR/generalization_13b.csv"
}

run_beta() {
  for beta in 0.05 0.1 0.2 0.4 0.6; do
    run_batch --target-model "$TARGET_7B" --methods async_sd \
      --per-workload-limit "$SENSITIVITY_PER_WORKLOAD" --beta "$beta" \
      --run-tag "beta_${beta}" --output "$RESULTS_DIR/controller_beta.csv"
  done
}

run_weights() {
  run_batch --target-model "$TARGET_7B" --methods async_sd \
    --per-workload-limit "$SENSITIVITY_PER_WORKLOAD" \
    --run-tag "weights_1x" --output "$RESULTS_DIR/controller_weights.csv"
  for scale in 0.5 2; do
    run_batch --target-model "$TARGET_7B" --methods async_sd \
      --per-workload-limit "$SENSITIVITY_PER_WORKLOAD" \
      --lambda-sw "$(awk "BEGIN {print 0.02 * $scale}")" \
      --run-tag "lambda_sw_${scale}x" --output "$RESULTS_DIR/controller_weights.csv"
    run_batch --target-model "$TARGET_7B" --methods async_sd \
      --per-workload-limit "$SENSITIVITY_PER_WORKLOAD" \
      --lambda-cmp "$(awk "BEGIN {print 0.02 * $scale}")" \
      --run-tag "lambda_cmp_${scale}x" --output "$RESULTS_DIR/controller_weights.csv"
    run_batch --target-model "$TARGET_7B" --methods async_sd \
      --per-workload-limit "$SENSITIVITY_PER_WORKLOAD" \
      --lambda-tail "$(awk "BEGIN {print 0.04 * $scale}")" \
      --run-tag "lambda_tail_${scale}x" --output "$RESULTS_DIR/controller_weights.csv"
  done
}

run_network() {
  while read -r name bandwidth rtt_ms; do
    run_batch --target-model "$TARGET_7B" --methods vanilla_dssd,dynamic_dssd,async_sd \
      --per-workload-limit "$NETWORK_PER_WORKLOAD" --bandwidth "$bandwidth" --rtt-ms "$rtt_ms" \
      --method-order rotate --async-shadow-gamma-cap 1 \
      --run-tag "network_${name}_adaptive" --output "$RESULTS_DIR/network_tail.csv"
    run_batch --target-model "$TARGET_7B" --methods optimized_dssd --gamma 1 \
      --per-workload-limit "$NETWORK_PER_WORKLOAD" --bandwidth "$bandwidth" --rtt-ms "$rtt_ms" \
      --run-tag "network_${name}_tuned_fixed_g1" --output "$RESULTS_DIR/network_tail.csv"
  done <<'EOF'
good 100 20
moderate 25 50
poor 5 100
EOF
}

run_dynamic_trace() {
  if [[ ! -f "$LINK_TRACE_PATH" ]]; then
    echo "Generating the default trace from true_wifi_move.py"
    "$PYTHON" "$CODE_DIR/true_wifi_move.py"
  fi
  run_batch --target-model "$TARGET_7B" \
    --methods vanilla_dssd,optimized_dssd,dynamic_dssd,async_sd \
    --controller-link-mode instantaneous \
    --link-trace-path "$LINK_TRACE_PATH" --link-trace-speedup "$DYNAMIC_TRACE_SPEEDUP" \
    --run-tag "dynamic_trace_${DYNAMIC_TRACE_SPEEDUP}x" \
    --output "$RESULTS_DIR/dynamic_trace_final.csv"
  run_batch --target-model "$TARGET_7B" --methods optimized_dssd --gamma 1 \
    --controller-link-mode instantaneous \
    --link-trace-path "$LINK_TRACE_PATH" --link-trace-speedup "$DYNAMIC_TRACE_SPEEDUP" \
    --run-tag "dynamic_trace_${DYNAMIC_TRACE_SPEEDUP}x_tuned_fixed_g1" \
    --output "$RESULTS_DIR/dynamic_trace_final.csv"
}

run_dynamic_trace_causal_pilot() {
  if [[ ! -f "$LINK_TRACE_PATH" ]]; then
    echo "Generating the default trace from true_wifi_move.py"
    "$PYTHON" "$CODE_DIR/true_wifi_move.py"
  fi
  run_batch --target-model "$TARGET_7B" \
    --methods dynamic_dssd,async_sd --method-order rotate \
    --per-workload-limit 3 \
    --controller-link-mode causal_previous_round \
    --link-trace-path "$LINK_TRACE_PATH" --link-trace-speedup "$DYNAMIC_TRACE_SPEEDUP" \
    --run-tag "dynamic_trace_${DYNAMIC_TRACE_SPEEDUP}x_causal_pilot" \
    --output "$RESULTS_DIR/dynamic_trace_causal_pilot.csv"
}

run_dynamic_trace_causal() {
  if [[ ! -f "$LINK_TRACE_PATH" ]]; then
    echo "Generating the default trace from true_wifi_move.py"
    "$PYTHON" "$CODE_DIR/true_wifi_move.py"
  fi
  run_batch --target-model "$TARGET_7B" \
    --methods dynamic_dssd,async_sd --method-order rotate \
    --controller-link-mode causal_previous_round \
    --link-trace-path "$LINK_TRACE_PATH" --link-trace-speedup "$DYNAMIC_TRACE_SPEEDUP" \
    --run-tag "dynamic_trace_${DYNAMIC_TRACE_SPEEDUP}x_causal" \
    --output "$RESULTS_DIR/dynamic_trace_causal.csv"
}

case "$STAGE" in
  sanity)
    run_sanity
    finalize_results "$RESULTS_DIR/sanity.csv" 1
    ;;
  pilot)
    run_pilot
    finalize_results "$RESULTS_DIR/pilot_68m_7b.csv" 9
    ;;
  overlap_pilot)
    run_overlap_pilot
    finalize_results "$RESULTS_DIR/overlap_pilot_cpu_gpu.csv" 9
    ;;
  overlap_pilot80)
    run_overlap_pilot80
    finalize_results "$RESULTS_DIR/overlap_pilot80_cpu_gpu.csv" 9
    ;;
  contention_audit)
    run_contention_audit
    ;;
  shadow_policy_pilot80)
    run_shadow_policy_pilot80
    finalize_results "$RESULTS_DIR/shadow_policy_pilot80.csv" 9
    "$PYTHON" "$CODE_DIR/analyze_shadow_policy.py" \
      "$RESULTS_DIR/shadow_policy_pilot80.csv"
    ;;
  shadow_network_pilot)
    run_shadow_network_pilot
    finalize_results "$RESULTS_DIR/shadow_network_pilot.csv" 9
    "$PYTHON" "$CODE_DIR/analyze_shadow_network.py" \
      "$RESULTS_DIR/shadow_network_pilot.csv"
    ;;
  shadow_anchor_pilot)
    run_shadow_anchor_pilot
    finalize_results "$RESULTS_DIR/shadow_anchor_pilot.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_shadow_anchor.py" \
      "$RESULTS_DIR/shadow_anchor_pilot.csv"
    ;;
  shadow_final_pilot)
    run_shadow_final_pilot
    finalize_results "$RESULTS_DIR/shadow_final_pilot.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_shadow_final.py" \
      "$RESULTS_DIR/shadow_final_pilot.csv"
    ;;
  shadow_model_pair_pilot)
    run_shadow_model_pair_pilot
    finalize_results "$RESULTS_DIR/shadow_pair_68m_7b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    finalize_results "$RESULTS_DIR/shadow_pair_160m_7b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_shadow_model_pair.py" \
      "$RESULTS_DIR/shadow_pair_68m_7b.csv" \
      "$RESULTS_DIR/shadow_pair_160m_7b.csv"
    ;;
  shadow_cap_pilot)
    run_shadow_cap_pilot
    finalize_results "$RESULTS_DIR/shadow_cap_pilot.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_shadow_cap.py" \
      "$RESULTS_DIR/shadow_cap_pilot.csv"
    ;;
  model_pair_audit)
    run_model_pair_audit
    ;;
  model_pair_audit_160m)
    run_model_pair_audit_160m
    ;;
  pair_pilot_160m)
    run_pair_pilot_160m
    finalize_results "$RESULTS_DIR/pilot_160m_7b.csv" 9
    finalize_results "$RESULTS_DIR/pilot_160m_13b.csv" 9
    ;;
  low_gamma_pilot)
    run_low_gamma_pilot
    finalize_results "$RESULTS_DIR/low_gamma_68m_7b.csv" 9
    finalize_results "$RESULTS_DIR/low_gamma_68m_13b.csv" 9
    finalize_results "$RESULTS_DIR/low_gamma_160m_7b.csv" 9
    finalize_results "$RESULTS_DIR/low_gamma_160m_13b.csv" 9
    ;;
  decision_pilot)
    run_decision_pilot
    finalize_results "$RESULTS_DIR/decision_pilot_68m_7b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_decision_pilot.py" \
      "$RESULTS_DIR/decision_pilot_68m_7b_summary.csv"
    ;;
  decision_pilot_13b)
    run_decision_pilot_13b
    finalize_results "$RESULTS_DIR/decision_pilot_68m_13b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_decision_pilot.py" \
      "$RESULTS_DIR/decision_pilot_68m_13b_summary.csv"
    ;;
  controller_fix_pilot)
    run_controller_fix_pilot
    finalize_results "$RESULTS_DIR/controller_fix_pilot.csv" "$((DECISION_PER_WORKLOAD * 3))"
    ;;
  controller_network_pilot)
    run_controller_network_pilot
    finalize_results "$RESULTS_DIR/controller_network_pilot.csv" 9
    ;;
  main)
    run_main
    finalize_results "$RESULTS_DIR/main_68m_7b.csv" 300
    ;;
  fixed_gamma)
    run_fixed_gamma
    finalize_results "$RESULTS_DIR/fixed_gamma.csv" 300
    ;;
  fixed_gamma_pilot)
    run_fixed_gamma_pilot
    finalize_results "$RESULTS_DIR/fixed_gamma_pilot_68m_7b.csv" 9
    finalize_results "$RESULTS_DIR/fixed_gamma_pilot_68m_13b.csv" 9
    finalize_results "$RESULTS_DIR/fixed_gamma_pilot_160m_7b.csv" 9
    finalize_results "$RESULTS_DIR/fixed_gamma_pilot_160m_13b.csv" 9
    ;;
  baselines)
    run_baselines
    finalize_results "$RESULTS_DIR/external_baselines.csv" 300
    ;;
  generalization)
    run_generalization
    finalize_results "$RESULTS_DIR/generalization_13b.csv" 300
    ;;
  beta)
    run_beta
    finalize_results "$RESULTS_DIR/controller_beta.csv" "$((SENSITIVITY_PER_WORKLOAD * 3))"
    ;;
  weights)
    run_weights
    finalize_results "$RESULTS_DIR/controller_weights.csv" "$((SENSITIVITY_PER_WORKLOAD * 3))"
    ;;
  network)
    run_network
    finalize_results "$RESULTS_DIR/network_tail.csv" "$((NETWORK_PER_WORKLOAD * 3))"
    ;;
  dynamic_trace)
    run_dynamic_trace
    finalize_results "$RESULTS_DIR/dynamic_trace_final.csv" 300
    ;;
  dynamic_trace_causal_pilot)
    run_dynamic_trace_causal_pilot
    finalize_results "$RESULTS_DIR/dynamic_trace_causal_pilot.csv" 9
    ;;
  dynamic_trace_causal)
    run_dynamic_trace_causal
    finalize_results "$RESULTS_DIR/dynamic_trace_causal.csv" 300
    ;;
  all)
    run_sanity
    finalize_results "$RESULTS_DIR/sanity.csv" 1
    run_pilot
    finalize_results "$RESULTS_DIR/pilot_68m_7b.csv" 9
    run_decision_pilot
    finalize_results "$RESULTS_DIR/decision_pilot_68m_7b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_decision_pilot.py" \
      "$RESULTS_DIR/decision_pilot_68m_7b_summary.csv"
    run_decision_pilot_13b
    finalize_results "$RESULTS_DIR/decision_pilot_68m_13b.csv" "$((DECISION_PER_WORKLOAD * 3))"
    "$PYTHON" "$CODE_DIR/analyze_decision_pilot.py" \
      "$RESULTS_DIR/decision_pilot_68m_13b_summary.csv"
    run_main
    finalize_results "$RESULTS_DIR/main_68m_7b.csv" 300
    run_fixed_gamma
    finalize_results "$RESULTS_DIR/fixed_gamma.csv" 300
    run_generalization
    finalize_results "$RESULTS_DIR/generalization_13b.csv" 300
    run_beta
    finalize_results "$RESULTS_DIR/controller_beta.csv" "$((SENSITIVITY_PER_WORKLOAD * 3))"
    run_weights
    finalize_results "$RESULTS_DIR/controller_weights.csv" "$((SENSITIVITY_PER_WORKLOAD * 3))"
    run_network
    finalize_results "$RESULTS_DIR/network_tail.csv" "$((NETWORK_PER_WORKLOAD * 3))"
    run_dynamic_trace
    finalize_results "$RESULTS_DIR/dynamic_trace.csv" 300
    ;;
  *)
    echo "Usage: $0 {sanity|pilot|overlap_pilot|overlap_pilot80|contention_audit|shadow_policy_pilot80|shadow_network_pilot|shadow_anchor_pilot|shadow_final_pilot|shadow_model_pair_pilot|shadow_cap_pilot|model_pair_audit|model_pair_audit_160m|pair_pilot_160m|low_gamma_pilot|fixed_gamma_pilot|decision_pilot|decision_pilot_13b|controller_fix_pilot|controller_network_pilot|main|baselines|fixed_gamma|generalization|beta|weights|network|dynamic_trace|dynamic_trace_causal_pilot|dynamic_trace_causal|all}" >&2
    exit 2
    ;;
esac
