#!/usr/bin/env python3
"""Check revision coverage, code provenance, and controller diagnostics."""

import argparse
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path


VARIANTS = {
    ("ratio", "legacy", "legacy"),
    ("difference", "legacy", "legacy"),
    ("ratio", "history", "legacy"),
    ("difference", "history", "legacy"),
    ("difference", "history", "lcb"),
}


def current_code_hash():
    digest = hashlib.sha256()
    for name in ("batch_experiments.py", "dynamic_dssd_main.py", "external_baselines.py", "pipesd_utils.py", "speculative.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def check(path, expected_prompts, methods):
    with open(path, newline="", encoding="utf-8") as handle:
        latest = {}
        for row in csv.DictReader(handle):
            latest[(row["config_id"], row["prompt_id"], row["method"])] = row
    rows = list(latest.values())
    expected_rows = expected_prompts * len(methods) * len(VARIANTS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    if {r["code_sha256"] for r in rows} != {current_code_hash()}:
        raise ValueError("CSV code hash does not match this checkout; use a fresh results directory")
    groups = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            raise ValueError(f"Unresolved error: {row['prompt_id']} {row['error']}")
        variant = tuple(row[key] for key in ("controller_score", "compression_estimator", "acceptance_predictor"))
        groups[(row["method"], variant)].append(row)
        if variant[0] == "difference":
            eta = float(row["controller_throughput_estimate_final"])
            if not math.isfinite(eta) or eta <= 0:
                raise ValueError("Invalid throughput estimate")
        if variant[1] == "history":
            count = int(row["controller_compression_observations"])
            pairs = int(row["controller_compression_history_pairs"])
            rejects = int(row["reject_rounds"])
            if not 0 <= pairs <= count <= rejects:
                raise ValueError("Invalid compression history accounting")
            if count:
                brier = float(row["controller_compression_brier_mean"])
                if not math.isfinite(brier) or not 0 <= brier <= 1:
                    raise ValueError("Invalid pre-update Brier score")
    if set(groups) != {(method, variant) for method in methods for variant in VARIANTS}:
        raise ValueError("Missing or unexpected controller/method combinations")
    prompt_sets = []
    for (method, variant), group in sorted(groups.items()):
        prompts = {(r["prompt_id"], r["input_hash"]) for r in group}
        if len(group) != expected_prompts or len(prompts) != expected_prompts:
            raise ValueError("Incomplete or duplicated prompt group")
        prompt_sets.append(prompts)
        seconds = sum(float(r["total_time_s"]) for r in group)
        tokens = sum(int(r["generated_tokens"]) for r in group)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid total runtime")
        observations = sum(int(r.get("controller_compression_observations") or 0) for r in group)
        print(f"{method:12s} {'/'.join(variant):28s} n={len(group)} "
              f"throughput={tokens / seconds:.3f} token/s history_n={observations}")
    if any(s != prompt_sets[0] for s in prompt_sets):
        raise ValueError("Variants did not evaluate identical prompts")
    print(f"[OK] {len(rows)} rows: provenance, coverage, and controller diagnostics verified")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--estimate-pilot", action="store_true")
    args = parser.parse_args()
    rows = check(args.csv_path, args.expected_prompts, set(args.methods.split(",")))
    if args.estimate_pilot:
        samples = [float(r["total_time_s"]) / int(r["generated_tokens"])
                   for r in rows if r["method"] == "async_sd"]
        if samples:
            estimate = 300 * 80 * sum(samples) / len(samples) / 60
            print(f"Pilot planning estimate: {estimate:.0f}-{2 * estimate:.0f} min plus model loading. "
                  "Scaled from short sanity runs; poor-link costs and long-prefix effects may exceed this range.")


if __name__ == "__main__":
    main()
