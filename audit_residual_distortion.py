"""Audit reject-residual distortion in a separate, non-timing experiment.

Pass ordinary batch_experiments.py options after --audit-output. This wrapper
temporarily retains full target probabilities inside in-process reject packets;
it must not be used to report throughput or latency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

def distribution_after_resample(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, bool]:
    residual = (p - q).clamp(min=0)
    if float(residual.sum()) > 1e-12:
        return residual / residual.sum(), False
    if float(p.sum()) > 1e-12:
        return p / p.sum(), True
    point = torch.zeros_like(q)
    point[int(q.argmax())] = 1
    return point, True


def tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p - q).abs().sum()) / 2.0


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "above_0.01_count": sum(value > 0.01 for value in values),
        "above_0.01_rate": sum(value > 0.01 for value in values) / len(values) if values else None,
        "above_0.05_count": sum(value > 0.05 for value in values),
        "above_0.05_rate": sum(value > 0.05 for value in values) / len(values) if values else None,
    }


def self_test() -> None:
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.2, 0.4, 0.4])
    full, _ = distribution_after_resample(p, q)
    sparse, fallback = distribution_after_resample(torch.tensor([0.6, 0.0, 0.0]), q)
    assert not fallback
    assert abs(tv(full, sparse)) < 1e-6
    _, fallback = distribution_after_resample(torch.tensor([0.0, 0.3, 0.0]), q)
    assert fallback
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.3, 0.4, 0.1])
    full, _ = distribution_after_resample(p, q)
    truncated, fallback = distribution_after_resample(torch.tensor([0.5, 0.0, 0.0]), q)
    assert not fallback
    assert abs(tv(full, truncated) - 1.0 / 3.0) < 1e-6
    assert summarize([0.0, 0.02])["above_0.01_count"] == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args, batch_args = parser.parse_known_args()
    if args.self_test:
        self_test()
        print("Residual audit self-test passed")
        return
    if not batch_args:
        parser.error("pass batch_experiments.py arguments after --audit-output")
    if "--output" not in batch_args or batch_args[-1] == "--output":
        parser.error("pass a fresh --output CSV path for the instrumented batch")
    batch_output = Path(batch_args[batch_args.index("--output") + 1])
    if batch_output.exists() and batch_output.stat().st_size:
        parser.error(f"audit batch output already contains data: {batch_output}")

    import batch_experiments
    import dynamic_dssd_main as core

    records = {
        "ideal_sparse_tv": [],
        "conditional_ideal_sparse_tv": [],
        "fp16_incremental_tv": [],
        "full_fp16_tv": [],
        "implemented_tv": [],
        "compressed_vs_full_fp16_tv": [],
    }
    counts = {"compressed": 0, "full": 0, "zero_full_residual": 0,
              "zero_sparse_residual": 0, "numerical_fallback": 0}
    original_build = core.build_reject_packet
    original_resample = core.UAVNode.resample_DSSD

    def audited_build(pj, packet_args):
        packet = original_build(pj, packet_args)
        packet["_audit_full_p"] = pj.detach().to(torch.float32).cpu().clone()
        return packet

    def audited_resample(node, j, packet, q_probs):
        if "_audit_full_p" in packet:
            full_p = packet["_audit_full_p"]
            q = q_probs[j - 1].detach().to(torch.float32).cpu()
            full_res = (full_p - q).clamp(min=0)
            counts[packet["mode"]] += 1
            if packet["mode"] == "compressed":
                ids = packet["ids"].long()
                ideal_p = torch.zeros_like(full_p)
                ideal_p[ids] = full_p[ids]
                actual_p = core.reconstruct_pj_from_packet(packet)
                ideal_dist, ideal_fallback = distribution_after_resample(ideal_p, q)
                actual_dist, actual_fallback = distribution_after_resample(actual_p, q)
                counts["zero_sparse_residual"] += int(ideal_fallback)
                counts["numerical_fallback"] += int(actual_fallback)
            if float(full_res.sum()) <= 1e-12:
                counts["zero_full_residual"] += 1
            elif packet["mode"] == "compressed":
                full_dist = full_res / full_res.sum()
                full_fp16_p = full_p.to(packet["probs"].dtype).to(torch.float32)
                full_fp16_dist, _ = distribution_after_resample(full_fp16_p, q)
                records["ideal_sparse_tv"].append(tv(full_dist, ideal_dist))
                if not ideal_fallback:
                    records["conditional_ideal_sparse_tv"].append(tv(full_dist, ideal_dist))
                records["fp16_incremental_tv"].append(tv(ideal_dist, actual_dist))
                records["full_fp16_tv"].append(tv(full_dist, full_fp16_dist))
                records["implemented_tv"].append(tv(full_dist, actual_dist))
                records["compressed_vs_full_fp16_tv"].append(tv(full_fp16_dist, actual_dist))
        return original_resample(node, j, packet, q_probs)

    core.build_reject_packet = audited_build
    core.UAVNode.resample_DSSD = audited_resample
    old_argv = sys.argv
    sys.argv = ["batch_experiments.py", *batch_args]
    completed = False
    try:
        batch_experiments.main()
        completed = True
    finally:
        sys.argv = old_argv
        core.build_reject_packet = original_build
        core.UAVNode.resample_DSSD = original_resample
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "completed": completed,
            "warning": "Audit instrumentation changes timing; do not use throughput/latency from this run.",
            "counts": counts,
            "tv": {name: summarize(values) for name, values in records.items()},
        }
        args.audit_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
