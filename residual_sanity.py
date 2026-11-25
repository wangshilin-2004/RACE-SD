#!/usr/bin/env python3
"""Opt-in numerical-residual audit; never use its timings as benchmark results."""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


THRESHOLD = 1e-12  # Must match UAVNode.resample_DSSD's existing branch.
MODES = ("compressed", "full")
COUNTS = ("calls", "fallback", "exact_zero", "small_positive", "argmax")
SUPPORTED_METHODS = {"vanilla_dssd", "optimized_dssd", "dynamic_dssd", "async_sd"}


class ResidualAudit:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {mode: dict.fromkeys(COUNTS, 0) for mode in MODES}
        self.examples = []

    def observe(self, mode, residual_mass, target_mass, metadata=None):
        if mode not in MODES:
            raise ValueError(f"Unknown reject packet mode: {mode}")
        if not all(math.isfinite(x) and x >= 0 for x in (residual_mass, target_mass)):
            raise ValueError("Nonfinite or negative residual/target mass")
        counts = self.counts[mode]
        counts["calls"] += 1
        if residual_mass <= THRESHOLD:
            counts["fallback"] += 1
            counts["exact_zero" if residual_mass == 0 else "small_positive"] += 1
            counts["argmax"] += int(target_mass <= THRESHOLD)
            if len(self.examples) < 5:
                self.examples.append(dict(
                    metadata or {}, mode=mode, residual_mass=residual_mass,
                    received_target_mass=target_mass,
                ))

    def fields(self):
        fields = {}
        for mode, counts in self.counts.items():
            for key, value in counts.items():
                fields[f"residual_{mode}_{key}"] = value
            fields[f"residual_{mode}_fallback_rate"] = (
                counts["fallback"] / counts["calls"] if counts["calls"] else ""
            )
        fields["residual_fallback_examples_json"] = json.dumps(self.examples, sort_keys=True)
        return fields


@contextmanager
def instrument(batch):
    """Wrap existing sampling read-only, with per-prompt counters and no RNG draws."""
    import torch

    core = batch.core
    original_resample = core.UAVNode.resample_DSSD
    original_run = batch.run_method
    original_row = batch.make_row
    audit = ResidualAudit()
    audit_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def resample(node, j, reject_packet, q_probs):
        with torch.no_grad():
            q = q_probs[j - 1].to(torch.float32).cpu()
            p = core.reconstruct_pj_from_packet(reject_packet)
            mass = float((p - q).clamp(min=0).sum().item())
            audit.observe(reject_packet["mode"], mass, float(p.sum().item()), {
                "rejection_position": int(j),
                "kept_mass": float(reject_packet.get("kept_mass", 1.0)),
                "packet_entries": int(reject_packet["probs"].numel()),
            })
        return original_resample(node, j, reject_packet, q_probs)

    def run(method, *args, **kwargs):
        audit.reset()
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Residual audit does not support {method}")
        metrics, stats = original_run(method, *args, **kwargs)
        for mode, stat in (("compressed", "reject_compressed_cnt"), ("full", "reject_full_cnt")):
            if audit.counts[mode]["calls"] != stats.get(stat, 0):
                raise RuntimeError(f"Residual audit count disagrees with {stat}")
        return metrics, stats

    def make_row(*args, **kwargs):
        row = original_row(*args, **kwargs)
        row.update(audit.fields())
        row["diagnostic_only"] = 1
        row["residual_threshold"] = THRESHOLD
        row["residual_audit_sha256"] = audit_hash
        return row

    core.UAVNode.resample_DSSD = resample
    batch.run_method = run
    batch.make_row = make_row
    try:
        yield audit
    finally:
        core.UAVNode.resample_DSSD = original_resample
        batch.run_method = original_run
        batch.make_row = original_row


def summarize(path):
    path = Path(path)
    groups = defaultdict(lambda: dict.fromkeys(COUNTS + ("prompt_runs", "affected_prompt_runs"), 0))
    errors = 0
    rows_seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("diagnostic_only") != "1":
                raise ValueError("Not a residual diagnostic CSV")
            if row["status"] != "ok":
                errors += 1
                continue
            identity = (row["config_id"], row["prompt_id"], row["method"])
            if identity in rows_seen:
                raise ValueError(f"Duplicate prompt run: {identity}")
            rows_seen.add(identity)
            for workload in (row["workload"], "ALL"):
                for mode in MODES:
                    key = (row["target_model"], row["method"], workload, mode)
                    group = groups[key]
                    group["prompt_runs"] += 1
                    group["affected_prompt_runs"] += int(int(row[f"residual_{mode}_fallback"]) > 0)
                    for name in COUNTS:
                        group[name] += int(row[f"residual_{mode}_{name}"])
    if not groups:
        raise ValueError("No successful diagnostic rows")
    out = path.with_name(path.stem + "_residual_summary.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["target_model", "method", "workload", "packet_mode", *COUNTS,
                  "prompt_runs", "affected_prompt_runs", "fallback_rate", "error_rows"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, counts in sorted(groups.items()):
            rate = counts["fallback"] / counts["calls"] if counts["calls"] else ""
            writer.writerow(dict(zip(fields[:4], key), **counts,
                                 fallback_rate=rate, error_rows=errors))
            if key[2] == "ALL":
                display = f"{rate:.6%}" if rate != "" else "N/A (no packets)"
                print(f"{key[1]:16s} {key[3]:10s}: {counts['fallback']}/{counts['calls']} "
                      f"({display}); affected prompt runs {counts['affected_prompt_runs']}/{counts['prompt_runs']}")
    print(f"Summary: {out}; successful runs: {len(rows_seen)}; error rows: {errors}")
    return out


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
        return
    import batch_experiments as batch

    cli = batch.parse_args()
    unknown = set(cli.methods.split(",")) - SUPPORTED_METHODS
    if unknown:
        raise SystemExit(f"Unsupported audit methods: {sorted(unknown)}")
    output = Path(cli.output)
    if output.exists():
        raise SystemExit("Audit output already exists; use a fresh path (no resume/overwrite).")
    cli.run_tag = "residual_audit_" + (cli.run_tag or "sanity")
    cli.no_resume = True
    original_parse = batch.parse_args
    batch.parse_args = lambda: cli
    print("DIAGNOSTIC ONLY: extra probability reductions can affect timing-based control. "
          "Do not use these throughput/latency columns as paper performance results.")
    try:
        with instrument(batch):
            batch.main()
    finally:
        batch.parse_args = original_parse
    summarize(output)


if __name__ == "__main__":
    main()
