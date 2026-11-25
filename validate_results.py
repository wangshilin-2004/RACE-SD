#!/usr/bin/env python3
"""Validate completeness and accounting invariants in batch experiment CSVs."""

import argparse
import csv
import json
import math
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Async-SD result CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--expected-prompts", type=int, default=0)
    return parser.parse_args()


def as_float(row, key, default=None):
    raw = row.get(key, "")
    return default if raw in (None, "") else float(raw)


def as_int(row, key, default=0):
    value = as_float(row, key, None)
    return default if value is None else int(value)


def main():
    args = parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise SystemExit("[FAIL] no rows found")

    failures = []
    warnings = []
    latest = {}
    for row in raw_rows:
        key = (row.get("config_id"), row.get("prompt_id"), row.get("method"))
        latest[key] = row
    rows = list(latest.values())
    superseded = len(raw_rows) - len(rows)
    if superseded:
        warnings.append(f"ignored {superseded} superseded retry row(s)")

    error_rows = [r for r in rows if r.get("status") != "ok"]
    if error_rows:
        failures.append(f"{len(error_rows)} rows have non-ok status")

    group_counts = Counter((r.get("config_id"), r.get("method")) for r in rows if r.get("status") == "ok")
    if args.expected_prompts:
        incomplete = {key: count for key, count in group_counts.items() if count != args.expected_prompts}
        if incomplete:
            failures.append(f"incomplete groups: {incomplete}")

    for row in rows:
        if row.get("status") != "ok":
            continue
        label = f"{row.get('method')}:{row.get('prompt_id')}"
        if not row.get("output_hash"):
            failures.append(f"{label} missing output hash")
        for key in ("throughput_tok_s", "total_time_s", "acceptance_rate"):
            value = as_float(row, key, None)
            if value is not None and (not math.isfinite(value) or value < 0):
                failures.append(f"{label} invalid {key}={value}")
        for key in (
            "preview_time_total_s", "preview_time_per_fresh_round_ms",
            "controller_decision_time_total_s", "controller_decision_time_per_round_ms",
        ):
            value = as_float(row, key, None)
            if value is not None and (not math.isfinite(value) or value < 0):
                failures.append(f"{label} invalid {key}={value}")
        acceptance = as_float(row, "acceptance_rate", None)
        if acceptance is not None and not 0 <= acceptance <= 1:
            failures.append(f"{label} acceptance_rate outside [0,1]")
        for key in (
            "avg_kept_mass", "avg_tail_mass",
            "avg_candidate_kept_mass", "avg_candidate_tail_mass",
        ):
            value = as_float(row, key, None)
            if value is not None and not 0 <= value <= 1:
                failures.append(f"{label} {key} outside [0,1]")
        candidate_samples = as_int(row, "candidate_mass_samples")
        if candidate_samples > 0:
            candidate_total = (
                as_float(row, "avg_candidate_kept_mass", 0.0)
                + as_float(row, "avg_candidate_tail_mass", 0.0)
            )
            if abs(candidate_total - 1.0) > 1e-4:
                failures.append(f"{label} candidate kept/tail mass does not sum to one")

        device_draft = row.get("device_draft", "")
        device_target = row.get("device_target", "")
        hardware_mode = row.get("hardware_mode", "")
        if {device_draft.split(":")[0], device_target.split(":")[0]} == {"cpu", "cuda"}:
            if hardware_mode != "cpu_gpu_split":
                failures.append(f"{label} CPU/GPU devices mislabeled as {hardware_mode}")

        throughput = as_float(row, "throughput_tok_s", None)
        total_time = as_float(row, "total_time_s", None)
        requested_tokens = as_int(row, "max_len")
        if throughput is not None and total_time is not None and requested_tokens > 0:
            generated_estimate = throughput * total_time
            generated_tokens = as_int(row, "generated_tokens", round(generated_estimate))
            if generated_tokens != requested_tokens or abs(generated_estimate - requested_tokens) > 0.01:
                failures.append(
                    f"{label} generated tokens {generated_tokens} "
                    f"(estimate {generated_estimate:.3f}) "
                    f"!= max_len {requested_tokens}"
                )

        method = row.get("method")
        rounds = as_int(row, "rounds")
        switches = as_int(row, "controller_switch_count")
        if switches > rounds:
            failures.append(f"{label} controller switches exceed controller decisions")
        if method in {
            "dsd", "vanilla_dssd", "optimized_dssd", "confidence_lookahead_dssd",
            "dynamic_dssd", "async_sd", "pipesd",
        }:
            accept_all = as_int(row, "accept_all_rounds")
            rejects = as_int(row, "reject_rounds")
            if accept_all + rejects != rounds:
                failures.append(f"{label} accept_all + rejects != rounds")
            if method not in {"dsd", "pipesd"}:
                compressed = as_int(row, "reject_compressed_cnt")
                full = as_int(row, "reject_full_cnt")
                if compressed + full != rejects:
                    failures.append(f"{label} compressed + full rejects != reject rounds")
            raw_samples = row.get("round_total_latency_samples_ms", "")
            samples = json.loads(raw_samples) if raw_samples else []
            if len(samples) != rounds:
                failures.append(f"{label} total-latency samples {len(samples)} != rounds {rounds}")
            if method in {"dynamic_dssd", "async_sd"}:
                preview_fresh = as_int(row, "preview_fresh_count")
                preview_cached = as_int(row, "preview_cached_count")
                if preview_fresh + preview_cached != rounds:
                    failures.append(
                        f"{label} fresh + cached previews != controller rounds"
                    )
                link_mode = row.get("controller_link_mode", "instantaneous")
                if link_mode not in {"instantaneous", "causal_previous_round"}:
                    failures.append(f"{label} unknown controller link mode {link_mode!r}")
                if link_mode == "causal_previous_round":
                    link_observations = as_int(row, "controller_link_observation_count")
                    if link_observations != rounds:
                        failures.append(
                            f"{label} causal link observations != controller rounds"
                        )

        if method in {
            "dsd", "vanilla_dssd", "optimized_dssd", "confidence_lookahead_dssd",
            "dynamic_dssd", "async_sd", "tkslt_as2", "pipesd",
        }:
            usage = json.loads(row.get("gamma_usage") or "{}")
            if sum(int(v) for v in usage.values()) != rounds:
                failures.append(f"{label} gamma usage count != rounds")

        if method == "pipesd":
            launches = as_int(row, "pipesd_waiting_launches")
            hits = as_int(row, "pipesd_waiting_hits")
            discard_reject = as_int(row, "pipesd_waiting_discard_reject")
            discard_mismatch = as_int(row, "pipesd_waiting_discard_mismatch")
            if hits + discard_reject + discard_mismatch != launches:
                failures.append(f"{label} PipeSD waiting outcomes != launches")
            rollback = as_int(row, "pipesd_rollback_tokens")
            waiting_tokens = as_int(row, "pipesd_waiting_tokens")
            if rollback > waiting_tokens:
                failures.append(f"{label} PipeSD rollback tokens exceed waiting tokens")

        if method == "tkslt_as2":
            accept_all = as_int(row, "accept_all_rounds")
            rejects = as_int(row, "reject_rounds")
            fallback = as_int(row, "tkslt_as2_fallback_rounds")
            if accept_all + rejects + fallback != rounds:
                failures.append(
                    f"{label} accept_all + rejects + AS2 fallback != rounds"
                )
            raw_samples = row.get("round_total_latency_samples_ms", "")
            samples = json.loads(raw_samples) if raw_samples else []
            if len(samples) != rounds:
                failures.append(
                    f"{label} total-latency samples {len(samples)} != rounds {rounds}"
                )

        if method == "async_sd":
            launches = as_int(row, "async_shadow_launches")
            hits = as_int(row, "async_shadow_hits")
            misses = as_int(row, "async_shadow_misses")
            outcomes = hits + misses
            if outcomes != launches:
                failures.append(f"{label} shadow outcomes {outcomes} != launches {launches}")
            explained_misses = (
                as_int(row, "async_shadow_discard_reject")
                + as_int(row, "async_shadow_discard_mismatch")
                + as_int(row, "async_shadow_discard_gamma_change")
                + as_int(row, "async_shadow_discard_prefix_mismatch")
                + as_int(row, "async_shadow_discard_generation_end")
            )
            if explained_misses > misses:
                failures.append(f"{label} shadow discard causes exceed total misses")
            partial_extensions = as_int(row, "async_shadow_partial_extensions")
            if partial_extensions > hits:
                failures.append(f"{label} partial shadow extensions exceed hits")
            eligible_raw = row.get("async_shadow_eligible_rounds", "")
            if eligible_raw not in (None, ""):
                eligible = as_int(row, "async_shadow_eligible_rounds")
                blocked_overlap = as_int(row, "async_shadow_blocked_overlap")
                blocked_benefit = as_int(row, "async_shadow_blocked_benefit")
                if eligible != launches + blocked_overlap + blocked_benefit:
                    failures.append(
                        f"{label} eligible shadow rounds do not reconcile with launch blocks"
                    )
            accept_all_launches = as_int(row, "async_shadow_accept_all_launches")
            anchor_matches = as_int(row, "async_shadow_anchor_matches")
            cancelled = as_int(row, "async_shadow_cancelled")
            if not 0 <= anchor_matches <= accept_all_launches <= launches:
                failures.append(f"{label} invalid accept-all/anchor shadow accounting")
            if cancelled > misses:
                failures.append(f"{label} cancelled shadows exceed total misses")
            launch_usage = json.loads(row.get("async_shadow_launch_gamma_usage") or "{}")
            if sum(int(v) for v in launch_usage.values()) != launches:
                failures.append(f"{label} shadow launch gamma usage != launches")
            total_compute = as_float(row, "async_shadow_total_time_s", 0.0)
            reused_compute = as_float(row, "async_shadow_reused_compute_time_s", 0.0)
            wasted_compute = as_float(row, "async_shadow_wasted_compute_time_s", 0.0)
            tolerance = max(1e-4, 1e-3 * max(total_compute, 1.0))
            if abs(total_compute - reused_compute - wasted_compute) > tolerance:
                failures.append(
                    f"{label} shadow compute accounting does not reconcile"
                )

    print(
        f"rows={len(rows)} raw_rows={len(raw_rows)} "
        f"successful={len(rows)-len(error_rows)} groups={len(group_counts)}"
    )
    for (config, method), count in sorted(group_counts.items()):
        print(f"  {method:16s} config={config} prompts={count}")
    for message in warnings:
        print(f"[WARN] {message}")
    for message in failures[:30]:
        print(f"[FAIL] {message}")
    if len(failures) > 30:
        print(f"[FAIL] ... and {len(failures)-30} more")
    if failures:
        return 1
    print("[ OK ] result accounting and completeness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
