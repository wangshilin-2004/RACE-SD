#!/usr/bin/env python3
"""Analyze short shadow-prefix caps under the final Async-SD implementation."""

import argparse
from collections import defaultdict

from analyze_shadow_anchor import latest_rows, paired_report, shadow_report
from analyze_shadow_model_pair import bootstrap_gain_interval


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the shadow gamma-cap pilot")
    parser.add_argument("csv_path")
    return parser.parse_args()


def main():
    groups = defaultdict(list)
    for row in latest_rows(parse_args().csv_path):
        groups[(row.get("run_tag", ""), row.get("method", ""))].append(row)

    print("Shadow gamma-cap diagnostic:")
    for cap in (1, 2, 8):
        run_tag = f"shadow_cap_{cap}"
        dynamic = groups.get((run_tag, "dynamic_dssd"), [])
        asynchronous = groups.get((run_tag, "async_sd"), [])
        print(f"\nshadow gamma cap={cap}")
        paired_report("  Async-SD vs Dynamic DSSD", dynamic, asynchronous)
        interval = bootstrap_gain_interval(dynamic, asynchronous)
        if interval is not None:
            print(
                "  paired gain bootstrap 95% interval="
                f"[{100.0 * interval[0]:+.2f}%, {100.0 * interval[1]:+.2f}%]"
            )
        shadow_report("  Async-SD shadow", asynchronous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
