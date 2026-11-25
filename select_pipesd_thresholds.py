#!/usr/bin/env python3
"""Rank PipeSD pilot configurations by aggregate generated-token throughput."""

import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    groups = defaultdict(list)
    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or row.get("method") != "pipesd":
                continue
            key = (
                float(row["pipesd_single_threshold"]),
                float(row["pipesd_multi_threshold"]),
            )
            groups[key].append(row)
    if not groups:
        raise SystemExit("No successful PipeSD pilot rows found")

    ranking = []
    for (single, multi), rows in groups.items():
        tokens = sum(float(row["generated_tokens"]) for row in rows)
        elapsed = sum(float(row["total_time_s"]) for row in rows)
        ranking.append((tokens / elapsed, single, multi, len(rows)))
    ranking.sort(reverse=True)
    print("PipeSD threshold ranking (aggregate throughput):")
    for throughput, single, multi, count in ranking:
        print(
            f"  single={single:.3f} multi={multi:.3f} "
            f"throughput={throughput:.3f} token/s prompts={count}"
        )
    best = ranking[0]
    print(
        f"SELECTED PIPE_SD_SINGLE={best[1]:.3f} "
        f"PIPE_SD_MULTI={best[2]:.3f}"
    )


if __name__ == "__main__":
    main()
