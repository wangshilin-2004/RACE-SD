#!/usr/bin/env python3
"""Paired report for the final Dynamic DSSD versus Async-SD pilot."""

import argparse
from collections import defaultdict

from analyze_shadow_anchor import latest_rows, paired_report, shadow_report


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the final shadow pilot")
    parser.add_argument("csv_path")
    return parser.parse_args()


def main():
    groups = defaultdict(list)
    for row in latest_rows(parse_args().csv_path):
        groups[row.get("method", "")].append(row)

    dynamic = groups.get("dynamic_dssd", [])
    asynchronous = groups.get("async_sd", [])
    print("Final shadow diagnostic:")
    paired_report("Async-SD vs Dynamic DSSD", dynamic, asynchronous)
    shadow_report("Async-SD", asynchronous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
