#!/usr/bin/env python3
"""Report whether a decision pilot is ready to proceed to full experiments."""

import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Summary CSV produced by summarize_results.py")
    return parser.parse_args()


def value(row, key):
    raw = row.get(key, "")
    return None if raw in (None, "") else float(raw)


def percentage(value_):
    return "n/a" if value_ is None else f"{100.0 * value_:.1f}%"


def method_label(row):
    if row["method"] == "optimized_dssd":
        return f"fixed gamma={int(float(row['gamma']))}"
    return row["method"]


def main():
    args = parse_args()
    with open(args.summary, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("workload") == "ALL"]

    required = {"vanilla_dssd", "dynamic_dssd", "async_sd", "optimized_dssd"}
    methods = {row.get("method") for row in rows}
    missing = sorted(required - methods)
    if missing:
        raise SystemExit(f"Decision summary is incomplete; missing: {', '.join(missing)}")

    rows.sort(key=lambda row: value(row, "throughput_aggregate_tok_s") or 0.0, reverse=True)
    print("\nDecision-pilot comparison (primary metric: aggregate throughput)")
    print("method                 aggregate   mean/prompt   p95(s)   accept   avg-gamma   safe")
    for row in rows:
        print(
            f"{method_label(row):<22}"
            f"{value(row, 'throughput_aggregate_tok_s'):>9.3f}   "
            f"{value(row, 'throughput_mean_tok_s'):>9.3f}   "
            f"{value(row, 'prompt_e2e_p95_s'):>6.3f}   "
            f"{percentage(value(row, 'acceptance_rate_weighted')):>7}   "
            f"{value(row, 'avg_gamma_mean'):>9.3f}   "
            f"{percentage(value(row, 'safe_mode_rate_weighted')):>6}"
        )

    fixed = [row for row in rows if row["method"] == "optimized_dssd"]
    best_fixed = max(fixed, key=lambda row: value(row, "throughput_aggregate_tok_s") or 0.0)
    dynamic = next(row for row in rows if row["method"] == "dynamic_dssd")
    async_row = next(row for row in rows if row["method"] == "async_sd")

    def relative(candidate, baseline):
        candidate_value = value(candidate, "throughput_aggregate_tok_s")
        baseline_value = value(baseline, "throughput_aggregate_tok_s")
        return candidate_value / baseline_value - 1.0

    print(
        f"\nBest fixed baseline: gamma={int(float(best_fixed['gamma']))}, "
        f"aggregate={value(best_fixed, 'throughput_aggregate_tok_s'):.3f} tok/s"
    )
    print(f"Dynamic vs best fixed: {relative(dynamic, best_fixed) * 100.0:+.2f}%")
    print(f"Async vs Dynamic:      {relative(async_row, dynamic) * 100.0:+.2f}%")
    launches = int(value(async_row, "shadow_launches") or 0)
    hits = int(value(async_row, "shadow_hits") or 0)
    reuse = value(async_row, "shadow_reuse_rate_weighted")
    print(f"Async shadow reuse:    {hits}/{launches} ({percentage(reuse)})")
    print(
        "\nThis report is diagnostic, not an automatic pass/fail decision. "
        "Inspect workload-level rows and logs before starting the full matrix."
    )


if __name__ == "__main__":
    main()
