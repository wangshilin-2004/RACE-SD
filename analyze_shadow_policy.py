#!/usr/bin/env python3
"""Paired diagnostic report for adaptive and aggressive shadow policies."""

import argparse
import csv
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Async-SD shadow policy pilots")
    parser.add_argument("csv_path")
    return parser.parse_args()


def number(row, key, default=0.0):
    raw = row.get(key, "")
    return default if raw in (None, "") else float(raw)


def latest_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest = {}
    for row in rows:
        key = (row.get("config_id"), row.get("prompt_id"), row.get("method"))
        latest[key] = row
    return [row for row in latest.values() if row.get("status") == "ok"]


def report_async(label, rows):
    if not rows:
        print(f"{label}: no rows")
        return
    launches = sum(number(row, "async_shadow_launches") for row in rows)
    hits = sum(number(row, "async_shadow_hits") for row in rows)
    rejects = sum(number(row, "async_shadow_discard_reject") for row in rows)
    mismatches = sum(number(row, "async_shadow_discard_mismatch") for row in rows)
    misses = sum(number(row, "async_shadow_misses") for row in rows)
    total_compute = sum(number(row, "async_shadow_total_time_s") for row in rows)
    wasted = sum(number(row, "async_shadow_wasted_compute_time_s") for row in rows)
    saved = sum(number(row, "async_shadow_saved_draft_time_s") for row in rows)
    wait = sum(number(row, "async_shadow_wait_time_s") for row in rows)
    throughput = statistics.mean(number(row, "throughput_tok_s") for row in rows)
    print(
        f"{label}: throughput={throughput:.3f} tok/s, "
        f"reuse={int(hits)}/{int(launches)} ({100.0 * hits / max(launches, 1):.1f}%), "
        f"discard reject/mismatch/miss={int(rejects)}/{int(mismatches)}/{int(misses)}"
    )
    print(
        f"  shadow compute={total_compute:.3f}s, wasted={wasted:.3f}s, "
        f"saved draft={saved:.3f}s, critical-path wait={wait:.3f}s, "
        f"saved-wait={saved - wait:+.3f}s"
    )


def main():
    rows = latest_rows(parse_args().csv_path)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("run_tag", ""), row.get("method", ""))].append(row)

    adaptive_dynamic = groups.get(("shadow_policy_adaptive", "dynamic_dssd"), [])
    adaptive_async = groups.get(("shadow_policy_adaptive", "async_sd"), [])
    aggressive_async = groups.get(("shadow_policy_always", "async_sd"), [])

    if adaptive_dynamic and adaptive_async:
        dynamic_by_prompt = {row["prompt_id"]: row for row in adaptive_dynamic}
        pairs = [
            (dynamic_by_prompt[row["prompt_id"]], row)
            for row in adaptive_async
            if row["prompt_id"] in dynamic_by_prompt
        ]
        gains = [
            number(async_row, "throughput_tok_s")
            / max(number(dynamic_row, "throughput_tok_s"), 1e-12)
            - 1.0
            for dynamic_row, async_row in pairs
        ]
        hash_matches = sum(
            dynamic_row.get("output_hash") == async_row.get("output_hash")
            for dynamic_row, async_row in pairs
        )
        print(
            f"Adaptive vs Dynamic: paired gain={100.0 * statistics.mean(gains):+.2f}%, "
            f"wins={sum(gain > 0 for gain in gains)}/{len(gains)}, "
            f"matching outputs={hash_matches}/{len(pairs)}"
        )
    else:
        print("Adaptive vs Dynamic: paired rows unavailable")

    report_async("Adaptive shadow", adaptive_async)
    report_async("Aggressive shadow", aggressive_async)
    if adaptive_dynamic and aggressive_async:
        dynamic_by_prompt = {row["prompt_id"]: row for row in adaptive_dynamic}
        pairs = [
            (dynamic_by_prompt[row["prompt_id"]], row)
            for row in aggressive_async
            if row["prompt_id"] in dynamic_by_prompt
        ]
        hash_matches = sum(
            dynamic_row.get("output_hash") == async_row.get("output_hash")
            for dynamic_row, async_row in pairs
        )
        print(f"Aggressive matching outputs={hash_matches}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
