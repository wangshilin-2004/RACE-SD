#!/usr/bin/env python3
"""Paired Async-SD shadow analysis across network conditions."""

import argparse
import csv
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze shadow network pilot")
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


def main():
    rows = latest_rows(parse_args().csv_path)
    conditions = defaultdict(list)
    for row in rows:
        conditions[(row.get("run_tag", ""), row.get("bandwidth_mbps", ""), row.get("rtt_ms", ""))].append(row)

    print("Shadow network diagnostic:")
    for (run_tag, bandwidth, rtt), group in sorted(conditions.items()):
        dynamic = {row["prompt_id"]: row for row in group if row["method"] == "dynamic_dssd"}
        async_rows = [row for row in group if row["method"] == "async_sd"]
        pairs = [
            (dynamic[row["prompt_id"]], row)
            for row in async_rows
            if row["prompt_id"] in dynamic
        ]
        if not pairs:
            continue
        gains = [
            number(async_row, "throughput_tok_s")
            / max(number(dynamic_row, "throughput_tok_s"), 1e-12)
            - 1.0
            for dynamic_row, async_row in pairs
        ]
        launches = sum(number(row, "async_shadow_launches") for row in async_rows)
        hits = sum(number(row, "async_shadow_hits") for row in async_rows)
        saved = sum(number(row, "async_shadow_saved_draft_time_s") for row in async_rows)
        wait = sum(number(row, "async_shadow_wait_time_s") for row in async_rows)
        wasted = sum(number(row, "async_shadow_wasted_compute_time_s") for row in async_rows)
        output_matches = sum(
            dynamic_row.get("output_hash") == async_row.get("output_hash")
            for dynamic_row, async_row in pairs
        )
        print(
            f"  {run_tag} ({bandwidth} Mbps/{rtt} ms): "
            f"gain={100.0 * statistics.mean(gains):+.2f}%, "
            f"wins={sum(gain > 0 for gain in gains)}/{len(gains)}, "
            f"reuse={int(hits)}/{int(launches)} "
            f"({100.0 * hits / max(launches, 1):.1f}%), "
            f"saved-wait={saved - wait:+.3f}s, wasted={wasted:.3f}s, "
            f"matching outputs={output_matches}/{len(pairs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
