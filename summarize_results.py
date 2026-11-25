#!/usr/bin/env python3
"""Aggregate per-prompt Async-SD CSV files into paper-ready summaries."""

import argparse
import csv
import json
import math
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Async-SD batch results")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def number(row, key):
    value = row.get(key, "")
    return None if value in (None, "") else float(value)


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else ""


def percentile(values, pct):
    values = sorted(float(x) for x in values)
    if not values:
        return ""
    rank = (len(values) - 1) * pct / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def ratio(rows, numerator, denominator):
    num = sum(number(row, numerator) or 0.0 for row in rows)
    den = sum(number(row, denominator) or 0.0 for row in rows)
    return num / den if den else ""


def weighted_mean(rows, value_key, weight_key):
    weighted = 0.0
    total_weight = 0.0
    for row in rows:
        value = number(row, value_key)
        weight = number(row, weight_key)
        if value is None or weight is None:
            continue
        weighted += value * weight
        total_weight += weight
    return weighted / total_weight if total_weight else ""


def pooled_samples(rows, key):
    result = []
    for row in rows:
        raw = row.get(key, "")
        if raw:
            result.extend(float(x) for x in json.loads(raw))
    return result


def aggregate(rows, workload):
    first = rows[0]
    accepted_total = sum(number(row, "accepted_draft_tokens_est") or 0.0 for row in rows)
    rejected_verified = sum(number(row, "reject_rounds") or 0.0 for row in rows)
    output = {
        "run_tag": first["run_tag"],
        "config_id": first["config_id"],
        "method": first["method"],
        "method_order_mode": first.get("method_order_mode", ""),
        "workload": workload,
        "prompts": len(rows),
        "draft_model": first["draft_model"],
        "target_model": first["target_model"],
        "draft_model_dtype": first.get("draft_model_dtype", first.get("model_dtype", "")),
        "target_model_dtype": first.get("target_model_dtype", first.get("model_dtype", "")),
        "resolved_draft_dtype": first.get("resolved_draft_dtype", ""),
        "resolved_target_dtype": first.get("resolved_target_dtype", ""),
        "draft_model_parameter_bytes": first.get("draft_model_parameter_bytes", ""),
        "target_model_parameter_bytes": first.get("target_model_parameter_bytes", ""),
        "draft_quantization": first.get("draft_quantization", "none"),
        "resolved_quantized_engine": first.get("resolved_quantized_engine", ""),
        "quantized_linear_count": first.get("quantized_linear_count", ""),
        "draft_state_tensor_bytes": first.get("draft_state_tensor_bytes", ""),
        "torch_num_threads": first.get("torch_num_threads", ""),
        "torch_num_interop_threads": first.get("torch_num_interop_threads", ""),
        "seed": first["seed"],
        "gamma": first["gamma"],
        "dynamic_gamma_set": first["dynamic_gamma_set"],
        "dynamic_min_gamma": first.get("dynamic_min_gamma", ""),
        "lookahead_confidence_threshold": first.get("lookahead_confidence_threshold", ""),
        "lookahead_min_gamma": first.get("lookahead_min_gamma", ""),
        "lookahead_max_gamma": first.get("lookahead_max_gamma", ""),
        "tkslt_k": first.get("tkslt_k", ""),
        "tkslt_gamma_set": first.get("tkslt_gamma_set", ""),
        "tkslt_beta": first.get("tkslt_beta", ""),
        "pipesd_single_threshold": first.get("pipesd_single_threshold", ""),
        "pipesd_multi_threshold": first.get("pipesd_multi_threshold", ""),
        "pipesd_min_gamma": first.get("pipesd_min_gamma", ""),
        "pipesd_max_gamma": first.get("pipesd_max_gamma", ""),
        "pipesd_packet_dtype": first.get("pipesd_packet_dtype", ""),
        "async_shadow_launch_mode": first.get("async_shadow_launch_mode", ""),
        "async_shadow_anchor_mode": first.get("async_shadow_anchor_mode", ""),
        "async_shadow_overlap_margin": first.get("async_shadow_overlap_margin", ""),
        "beta": first["beta"],
        "controller_score": first.get("controller_score", "ratio"),
        "compression_estimator": first.get("compression_estimator", "legacy"),
        "acceptance_predictor": first.get("acceptance_predictor", "legacy"),
        "z_alpha": first.get("z_alpha", ""),
        "alpha_weight_history": first.get("alpha_weight_history", ""),
        "alpha_weight_difficulty": first.get("alpha_weight_difficulty", ""),
        "alpha_weight_early_reject": first.get("alpha_weight_early_reject", ""),
        "cmp_weight_mass": first.get("cmp_weight_mass", ""),
        "cmp_weight_saving": first.get("cmp_weight_saving", ""),
        "cmp_weight_topk": first.get("cmp_weight_topk", ""),
        "cmp_weight_easiness": first.get("cmp_weight_easiness", ""),
        "cmp_threshold_penalty": first.get("cmp_threshold_penalty", ""),
        "cmp_k_ref": first.get("cmp_k_ref", ""),
        "cmp_tau0": first.get("cmp_tau0", ""),
        "cmp_delta_tau": first.get("cmp_delta_tau", ""),
        "dynamic_reject_topk_set": first.get("dynamic_reject_topk_set", ""),
        "dynamic_reject_mass_set": first.get("dynamic_reject_mass_set", ""),
        "lambda_sw": first["lambda_sw"],
        "lambda_cmp": first["lambda_cmp"],
        "lambda_tail": first["lambda_tail"],
        "bandwidth_mbps": first["bandwidth_mbps"],
        "rtt_ms": first["rtt_ms"],
        "link_trace_sha256": first.get("link_trace_sha256", ""),
        "link_trace_speedup": first.get("link_trace_speedup", "1.0"),
        "throughput_mean_tok_s": mean(number(row, "throughput_tok_s") for row in rows),
        "throughput_aggregate_tok_s": (
            sum(number(row, "generated_tokens") or 0.0 for row in rows)
            / max(sum(number(row, "total_time_s") or 0.0 for row in rows), 1e-12)
        ),
        "prompt_e2e_mean_s": mean(number(row, "total_time_s") for row in rows),
        "slm_time_mean_s": mean(number(row, "total_slm_s") for row in rows),
        "e2e_time_total_s": sum(number(row, "total_time_s") or 0.0 for row in rows),
        "prompt_e2e_p50_s": percentile(
            [number(row, "total_time_s") for row in rows if number(row, "total_time_s") is not None], 50
        ),
        "prompt_e2e_p95_s": percentile(
            [number(row, "total_time_s") for row in rows if number(row, "total_time_s") is not None], 95
        ),
        "prompt_e2e_p99_s": percentile(
            [number(row, "total_time_s") for row in rows if number(row, "total_time_s") is not None], 99
        ),
        "acceptance_rate_weighted": ratio(rows, "accepted_draft_tokens_est", "proposed_tokens"),
        "verified_token_acceptance_rate": (
            accepted_total / (accepted_total + rejected_verified)
            if accepted_total + rejected_verified > 0.0 else ""
        ),
        "accept_all_round_rate": ratio(rows, "accept_all_rounds", "rounds"),
        "avg_gamma_mean": mean(number(row, "avg_gamma") for row in rows),
        "switch_rate_weighted": ratio(rows, "controller_switch_count", "rounds"),
        "safe_mode_rate_weighted": ratio(rows, "controller_safe_mode_count", "rounds"),
        "warmup_rate_weighted": ratio(rows, "controller_warmup_count", "rounds"),
        "hold_rate_weighted": ratio(rows, "controller_hold_count", "rounds"),
        "controller_alpha_ema_final_mean": mean(
            number(row, "controller_alpha_ema_final") for row in rows
        ),
        "controller_alpha_lcb_final_mean": mean(
            number(row, "controller_alpha_lcb_final") for row in rows
        ),
        "controller_reject_rate_ema_final_mean": mean(
            number(row, "controller_reject_rate_ema_final") for row in rows
        ),
        "controller_early_reject_ema_final_mean": mean(
            number(row, "controller_early_reject_ema_final") for row in rows
        ),
        "preview_time_total_s": sum(
            number(row, "preview_time_total_s") or 0.0 for row in rows
        ),
        "preview_fresh_count": sum(
            number(row, "preview_fresh_count") or 0.0 for row in rows
        ),
        "preview_cached_count": sum(
            number(row, "preview_cached_count") or 0.0 for row in rows
        ),
        "controller_decision_time_total_s": sum(
            number(row, "controller_decision_time_total_s") or 0.0 for row in rows
        ),
        "compression_on_ratio_mean": mean(number(row, "compression_on_ratio") for row in rows),
        "compression_hit_rate_weighted": ratio(rows, "reject_compressed_cnt", "reject_rounds"),
        "candidate_kept_mass_weighted": weighted_mean(
            rows, "avg_candidate_kept_mass", "candidate_mass_samples"
        ),
        "candidate_tail_mass_weighted": weighted_mean(
            rows, "avg_candidate_tail_mass", "candidate_mass_samples"
        ),
        "shadow_launch_rate_weighted": ratio(rows, "async_shadow_launches", "rounds"),
        "shadow_reuse_rate_weighted": ratio(rows, "async_shadow_hits", "async_shadow_launches"),
        "shadow_launches": sum(number(row, "async_shadow_launches") or 0 for row in rows),
        "shadow_hits": sum(number(row, "async_shadow_hits") or 0 for row in rows),
        "shadow_discard_reject": sum(number(row, "async_shadow_discard_reject") or 0 for row in rows),
        "shadow_discard_mismatch": sum(number(row, "async_shadow_discard_mismatch") or 0 for row in rows),
        "shadow_discard_gamma_change": sum(
            number(row, "async_shadow_discard_gamma_change") or 0 for row in rows
        ),
        "shadow_discard_prefix_mismatch": sum(
            number(row, "async_shadow_discard_prefix_mismatch") or 0 for row in rows
        ),
        "shadow_discard_generation_end": sum(
            number(row, "async_shadow_discard_generation_end") or 0 for row in rows
        ),
        "shadow_partial_extensions": sum(
            number(row, "async_shadow_partial_extensions") or 0 for row in rows
        ),
        "shadow_eligible_rounds": sum(
            number(row, "async_shadow_eligible_rounds") or 0 for row in rows
        ),
        "shadow_blocked_overlap": sum(
            number(row, "async_shadow_blocked_overlap") or 0 for row in rows
        ),
        "shadow_blocked_benefit": sum(
            number(row, "async_shadow_blocked_benefit") or 0 for row in rows
        ),
        "shadow_misses": sum(number(row, "async_shadow_misses") or 0 for row in rows),
        "shadow_accept_all_launches": sum(
            number(row, "async_shadow_accept_all_launches") or 0 for row in rows
        ),
        "shadow_anchor_matches": sum(
            number(row, "async_shadow_anchor_matches") or 0 for row in rows
        ),
        "shadow_cancelled": sum(number(row, "async_shadow_cancelled") or 0 for row in rows),
        "shadow_tokens_computed": sum(
            number(row, "async_shadow_tokens_computed") or 0 for row in rows
        ),
        "shadow_predicted_reuse_mean": mean(
            number(row, "async_shadow_predicted_reuse_mean") for row in rows
        ),
        "shadow_expected_benefit_mean_s": mean(
            number(row, "async_shadow_expected_benefit_mean_s") for row in rows
        ),
        "shadow_wasted_compute_s": sum(number(row, "async_shadow_wasted_compute_time_s") or 0 for row in rows),
        "shadow_reused_compute_s": sum(number(row, "async_shadow_reused_compute_time_s") or 0 for row in rows),
        "shadow_saved_draft_s": sum(number(row, "async_shadow_saved_draft_time_s") or 0 for row in rows),
        "shadow_wait_s": sum(number(row, "async_shadow_wait_time_s") or 0 for row in rows),
        "shadow_compute_per_token_ema_s_mean": mean(
            number(row, "async_shadow_compute_per_token_ema_s") for row in rows
        ),
        "tkslt_uplink_sparse_ratio_mean": mean(
            number(row, "tkslt_uplink_sparse_ratio") for row in rows
        ),
        "tkslt_as2_fallback_rate_weighted": ratio(
            rows, "tkslt_as2_fallback_rounds", "rounds"
        ),
        "pipesd_dp_batches_per_round_mean": mean(
            number(row, "pipesd_dp_batches_per_round") for row in rows
        ),
        "pipesd_uplink_service_s": sum(
            number(row, "pipesd_uplink_service_time_s") or 0.0 for row in rows
        ),
        "pipesd_uplink_hidden_s": sum(
            number(row, "pipesd_uplink_hidden_time_s") or 0.0 for row in rows
        ),
        "pipesd_waiting_launches": sum(
            number(row, "pipesd_waiting_launches") or 0 for row in rows
        ),
        "pipesd_waiting_hits": sum(
            number(row, "pipesd_waiting_hits") or 0 for row in rows
        ),
        "pipesd_waiting_reuse_rate": ratio(
            rows, "pipesd_waiting_hits", "pipesd_waiting_launches"
        ),
        "pipesd_rollback_tokens": sum(
            number(row, "pipesd_rollback_tokens") or 0 for row in rows
        ),
        "pipesd_waiting_overhang_s": sum(
            number(row, "pipesd_waiting_overhang_time_s") or 0.0 for row in rows
        ),
    }
    actual = sum(number(row, "reject_down_bytes_actual") or 0 for row in rows)
    full = sum(number(row, "reject_down_bytes_full_equiv") or 0 for row in rows)
    output["reject_downlink_saving"] = 1.0 - actual / full if full else ""
    output["preview_time_per_fresh_round_ms"] = (
        1000.0 * output["preview_time_total_s"] / output["preview_fresh_count"]
        if output["preview_fresh_count"] else ""
    )
    total_rounds = sum(number(row, "rounds") or 0.0 for row in rows)
    output["controller_decision_time_per_round_ms"] = (
        1000.0 * output["controller_decision_time_total_s"] / total_rounds
        if total_rounds else ""
    )
    output["controller_time_fraction_e2e"] = (
        output["controller_decision_time_total_s"] / output["e2e_time_total_s"]
        if output["e2e_time_total_s"] else ""
    )
    for component in ("total", "comm", "slm", "llm"):
        samples = pooled_samples(rows, f"round_{component}_latency_samples_ms")
        output[f"round_{component}_mean_ms"] = mean(samples)
        output[f"round_{component}_p50_ms"] = percentile(samples, 50)
        output[f"round_{component}_p95_ms"] = percentile(samples, 95)
        output[f"round_{component}_p99_ms"] = percentile(samples, 99)
        output[f"round_{component}_samples"] = len(samples)
    return output


def main():
    args = parse_args()
    raw_rows = []
    for path in args.inputs:
        with open(path, newline="", encoding="utf-8") as handle:
            raw_rows.extend(csv.DictReader(handle))
    latest = {}
    for row in raw_rows:
        key = (row.get("config_id"), row.get("prompt_id"), row.get("method"))
        latest[key] = row
    rows = [row for row in latest.values() if row.get("status") == "ok"]
    if not rows:
        raise SystemExit("No successful rows found")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["config_id"], row["method"], row["workload"])].append(row)
        groups[(row["config_id"], row["method"], "ALL")].append(row)
    summaries = [aggregate(group, key[2]) for key, group in sorted(groups.items())]
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} summary rows to {args.output}")
    print("\nOverall summary:")
    for row in summaries:
        if row["workload"] != "ALL":
            continue
        message = (
            f"  {row['method']:<27} "
            f"throughput={float(row['throughput_mean_tok_s']):.3f} tok/s  "
            f"aggregate={float(row['throughput_aggregate_tok_s']):.3f} tok/s  "
            f"mean={float(row['prompt_e2e_mean_s']):.3f}s  "
            f"p95={float(row['prompt_e2e_p95_s']):.3f}s"
        )
        if row["method"] == "async_sd" and row["shadow_launches"]:
            reuse = row["shadow_reuse_rate_weighted"]
            message += (
                f"  shadow={int(float(row['shadow_hits']))}/{int(float(row['shadow_launches']))}"
                f" ({float(reuse) * 100.0:.1f}%)"
                f"  mode={row['async_shadow_launch_mode']}"
            )
        print(message)


if __name__ == "__main__":
    main()
