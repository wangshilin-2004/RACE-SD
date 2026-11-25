#!/usr/bin/env python3
"""Compare Dynamic DSSD and Async-SD across candidate draft model pairs."""

import argparse
import os
import random
import statistics
from collections import defaultdict

from analyze_shadow_anchor import (
    aggregate_throughput,
    latest_rows,
    number,
    paired_report,
    shadow_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the shadow model-pair pilot")
    parser.add_argument("csv_paths", nargs="+")
    return parser.parse_args()


def weighted_acceptance(rows):
    accepted = sum(number(row, "accepted_draft_tokens_est") for row in rows)
    proposed = sum(number(row, "proposed_tokens") for row in rows)
    return accepted / max(proposed, 1.0)


def mean_gamma(rows):
    rounds = sum(number(row, "rounds") for row in rows)
    weighted = sum(number(row, "avg_gamma") * number(row, "rounds") for row in rows)
    return weighted / max(rounds, 1.0)


def model_label(rows, fallback):
    if not rows:
        return fallback
    draft = os.path.basename(rows[0].get("draft_model", "draft"))
    target = os.path.basename(rows[0].get("target_model", "target"))
    return f"{draft} -> {target}"


def bootstrap_gain_interval(baseline_rows, candidate_rows, samples=20000):
    baseline = {row["prompt_id"]: row for row in baseline_rows}
    gains = [
        number(row, "throughput_tok_s")
        / max(number(baseline[row["prompt_id"]], "throughput_tok_s"), 1e-12)
        - 1.0
        for row in candidate_rows
        if row["prompt_id"] in baseline
    ]
    if not gains:
        return None
    rng = random.Random(321)
    estimates = sorted(
        statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]


def main():
    pair_results = []
    print("Shadow model-pair diagnostic:")
    for path in parse_args().csv_paths:
        groups = defaultdict(list)
        for row in latest_rows(path):
            groups[row.get("method", "")].append(row)

        dynamic = groups.get("dynamic_dssd", [])
        asynchronous = groups.get("async_sd", [])
        label = model_label(dynamic or asynchronous, os.path.basename(path))
        print(f"\n{label}")
        paired_report("  Async-SD vs Dynamic DSSD", dynamic, asynchronous)
        interval = bootstrap_gain_interval(dynamic, asynchronous)
        if interval is not None:
            print(
                "  paired gain bootstrap 95% interval="
                f"[{100.0 * interval[0]:+.2f}%, {100.0 * interval[1]:+.2f}%]"
            )
        for name, rows in (("Dynamic DSSD", dynamic), ("Async-SD", asynchronous)):
            print(
                f"  {name}: throughput={aggregate_throughput(rows):.3f} tok/s, "
                f"acceptance={100.0 * weighted_acceptance(rows):.1f}%, "
                f"avg-gamma={mean_gamma(rows):.3f}"
            )
        shadow_report("  Async-SD shadow", asynchronous)
        pair_results.append((label, dynamic, asynchronous))

    if len(pair_results) >= 2:
        base_label, base_dynamic, base_async = pair_results[0]
        for label, dynamic, asynchronous in pair_results[1:]:
            dynamic_gain = aggregate_throughput(dynamic) / max(
                aggregate_throughput(base_dynamic), 1e-12
            ) - 1.0
            async_gain = aggregate_throughput(asynchronous) / max(
                aggregate_throughput(base_async), 1e-12
            ) - 1.0
            print(
                f"\nPair-level throughput ({label} vs {base_label}): "
                f"Dynamic={100.0 * dynamic_gain:+.2f}%, "
                f"Async={100.0 * async_gain:+.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
