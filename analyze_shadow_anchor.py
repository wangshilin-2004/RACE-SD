#!/usr/bin/env python3
"""Paired report for sampled- and MAP-anchor shadow drafting."""

import argparse
import csv
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the shadow-anchor pilot")
    parser.add_argument("csv_path")
    return parser.parse_args()


def number(row, key, default=0.0):
    raw = row.get(key, "")
    return default if raw in (None, "") else float(raw)


def latest_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    latest = {}
    for row in raw_rows:
        key = (row.get("config_id"), row.get("prompt_id"), row.get("method"))
        latest[key] = row
    return [row for row in latest.values() if row.get("status") == "ok"]


def aggregate_throughput(rows):
    tokens = sum(number(row, "generated_tokens") for row in rows)
    elapsed = sum(number(row, "total_time_s") for row in rows)
    return tokens / max(elapsed, 1e-12)


def paired_report(label, baseline_rows, candidate_rows):
    baseline = {row["prompt_id"]: row for row in baseline_rows}
    pairs = [
        (baseline[row["prompt_id"]], row)
        for row in candidate_rows
        if row["prompt_id"] in baseline
    ]
    if not pairs:
        print(f"{label}: paired rows unavailable")
        return

    gains = [
        number(candidate, "throughput_tok_s")
        / max(number(reference, "throughput_tok_s"), 1e-12)
        - 1.0
        for reference, candidate in pairs
    ]
    aggregate_gain = (
        aggregate_throughput(candidate_rows)
        / max(aggregate_throughput(baseline_rows), 1e-12)
        - 1.0
    )
    hash_matches = sum(
        reference.get("output_hash") == candidate.get("output_hash")
        for reference, candidate in pairs
    )
    print(
        f"{label}: aggregate gain={100.0 * aggregate_gain:+.2f}%, "
        f"mean paired gain={100.0 * statistics.mean(gains):+.2f}%, "
        f"wins={sum(gain > 0.0 for gain in gains)}/{len(gains)}, "
        f"matching outputs={hash_matches}/{len(pairs)}"
    )


def shadow_report(label, rows):
    if not rows:
        print(f"{label}: no rows")
        return
    launches = sum(number(row, "async_shadow_launches") for row in rows)
    hits = sum(number(row, "async_shadow_hits") for row in rows)
    causes = {
        "reject": sum(number(row, "async_shadow_discard_reject") for row in rows),
        "anchor": sum(number(row, "async_shadow_discard_mismatch") for row in rows),
        "gamma": sum(number(row, "async_shadow_discard_gamma_change") for row in rows),
        "prefix": sum(number(row, "async_shadow_discard_prefix_mismatch") for row in rows),
        "end": sum(number(row, "async_shadow_discard_generation_end") for row in rows),
    }
    saved = sum(number(row, "async_shadow_saved_draft_time_s") for row in rows)
    wait = sum(number(row, "async_shadow_wait_time_s") for row in rows)
    wasted = sum(number(row, "async_shadow_wasted_compute_time_s") for row in rows)
    partial = sum(number(row, "async_shadow_partial_extensions") for row in rows)
    eligible = sum(number(row, "async_shadow_eligible_rounds") for row in rows)
    blocked_overlap = sum(number(row, "async_shadow_blocked_overlap") for row in rows)
    blocked_benefit = sum(number(row, "async_shadow_blocked_benefit") for row in rows)
    print(
        f"{label}: aggregate throughput={aggregate_throughput(rows):.3f} tok/s, "
        f"reuse={int(hits)}/{int(launches)} "
        f"({100.0 * hits / max(launches, 1):.1f}%)"
    )
    print(
        "  discards reject/anchor/gamma/prefix/end="
        f"{int(causes['reject'])}/{int(causes['anchor'])}/{int(causes['gamma'])}/"
        f"{int(causes['prefix'])}/{int(causes['end'])}; "
        f"saved={saved:.3f}s, wait={wait:.3f}s, wasted={wasted:.3f}s, "
        f"saved-wait={saved - wait:+.3f}s"
    )
    print(
        f"  eligible={int(eligible)}, blocked overlap/benefit="
        f"{int(blocked_overlap)}/{int(blocked_benefit)}, "
        f"partial extensions={int(partial)}"
    )


def main():
    groups = defaultdict(list)
    for row in latest_rows(parse_args().csv_path):
        groups[(row.get("run_tag", ""), row.get("method", ""))].append(row)

    dynamic = groups.get(("shadow_anchor_dynamic", "dynamic_dssd"), [])
    sampled = groups.get(("shadow_anchor_sample", "async_sd"), [])
    mapped = groups.get(("shadow_anchor_map", "async_sd"), [])

    print("Shadow anchor diagnostic:")
    paired_report("Sample anchor vs Dynamic", dynamic, sampled)
    paired_report("MAP anchor vs Dynamic", dynamic, mapped)
    paired_report("MAP anchor vs Sample anchor", sampled, mapped)
    shadow_report("Sample anchor", sampled)
    shadow_report("MAP anchor", mapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
