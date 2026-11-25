#!/usr/bin/env python3
"""Preflight checks for the Async-SD prompt set and experiment environment."""

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_RESULT_COLUMNS = {
    "method", "draft_model", "target_model", "workload", "prompt_id", "seed",
    "gamma", "dynamic_gamma_set", "beta", "lambda_sw", "lambda_cmp", "lambda_tail",
    "bandwidth_mbps", "rtt_ms", "reject_topk", "mass_threshold", "async_enabled",
    "lookahead_confidence_threshold", "tkslt_k", "tkslt_gamma_set",
    "prompt_file_sha256", "code_sha256", "input_hash", "hardware_mode",
    "link_trace_sha256", "link_trace_speedup", "cuda_target_device_name", "torch_num_threads",
    "controller_link_mode",
    "throughput_tok_s", "acceptance_rate", "round_total_latency_p95_ms",
    "output_hash", "async_shadow_launch_mode", "async_shadow_anchor_mode",
    "async_shadow_overlap_margin",
    "method_order_mode", "execution_order_index",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Async-SD experiment inputs")
    parser.add_argument("--prompt-file", default="prompts_300.jsonl")
    parser.add_argument("--draft-model", default="")
    parser.add_argument("--target-model", default="")
    parser.add_argument("--device-draft", default="cpu")
    parser.add_argument("--device-target", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling-top-k", type=int, default=0)
    parser.add_argument("--reject-topk", type=int, default=64)
    parser.add_argument("--sampling-top-p", type=float, default=0.0)
    parser.add_argument("--check-csv", default="")
    parser.add_argument("--execute", action="store_true", help="Run one short Async-SD prompt")
    parser.add_argument("--output", default="results/sanity.csv")
    return parser.parse_args()


def fail(message):
    print(f"[FAIL] {message}")
    return 1


def ok(message):
    print(f"[ OK ] {message}")


def warn(message):
    print(f"[WARN] {message}")


def validate_prompts(path):
    errors = 0
    counts = collections.Counter()
    texts = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors += fail(f"invalid JSON at line {line_number}: {exc}")
                continue
            if set(row) != {"source", "text"}:
                errors += fail(f"line {line_number} has keys {sorted(row)}")
                continue
            if not str(row["text"]).strip():
                errors += fail(f"empty prompt at line {line_number}")
            counts[str(row["source"])] += 1
            texts.append(str(row["text"]))
    ok(f"prompt rows: {sum(counts.values())}; workloads: {dict(counts)}")
    duplicate_count = len(texts) - len(set(texts))
    if duplicate_count:
        warn(f"{duplicate_count} duplicate prompt texts found; IDs remain distinct by source index")
    if counts != collections.Counter({"alpaca": 100, "wmt14": 100, "gsm8k": 100}):
        warn("workload composition differs from the paper's 100/100/100 setup")
    return errors


def validate_model(label, value):
    if not value:
        warn(f"{label} model was not supplied; model loading was not checked")
        return 0
    if os.path.isabs(value) or value.startswith("."):
        if not os.path.exists(value):
            return fail(f"{label} model path does not exist: {value}")
        ok(f"{label} model path exists: {value}")
    else:
        ok(f"{label} model treated as a Hugging Face model ID: {value}")
    return 0


def validate_csv(path):
    if not path:
        return 0
    if not os.path.exists(path):
        return fail(f"CSV does not exist: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_RESULT_COLUMNS - set(reader.fieldnames or [])
        rows = list(reader)
    if missing:
        return fail(f"CSV is missing columns: {sorted(missing)}")
    ok(f"CSV schema is complete; rows: {len(rows)}")
    errors = sum(row.get("status") == "error" for row in rows)
    if errors:
        warn(f"CSV contains {errors} failed runs")
    return 0


def main():
    args = parse_args()
    errors = 0
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        errors += fail(f"prompt file does not exist: {prompt_path}")
    else:
        errors += validate_prompts(prompt_path)

    try:
        import torch
        import transformers
        ok(f"PyTorch {torch.__version__}; Transformers {transformers.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA devices: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                ok(f"cuda:{index} {props.name}; memory={props.total_memory / 2**30:.1f} GiB")
            if args.device_draft == args.device_target and args.device_draft.startswith("cuda"):
                warn("draft and target use the same GPU; async wall time does not represent two-GPU overlap")
            elif (
                args.device_draft.startswith("cuda")
                and args.device_target.startswith("cuda")
                and args.device_draft != args.device_target
                and torch.cuda.device_count() < 2
            ):
                errors += fail("two different CUDA devices requested, but fewer than two are visible")
            elif args.device_draft == "cpu" and args.device_target.startswith("cuda"):
                ok("CPU draft and GPU target use separate compute resources for overlap timing")
        else:
            warn("CUDA is unavailable; full experiments will be impractically slow")
    except ImportError as exc:
        errors += fail(f"missing Python dependency: {exc}")

    errors += validate_model("draft", args.draft_model)
    errors += validate_model("target", args.target_model)
    errors += validate_csv(args.check_csv)
    if args.sampling_top_k > 0 and args.reject_topk >= args.sampling_top_k:
        warn(
            "reject top-k is at least as large as sampling top-k; the target distribution "
            "has no more nonzero entries than the sparse packet and compression may become "
            "a degenerate comparison"
        )

    if args.execute:
        if errors or not args.draft_model or not args.target_model:
            errors += fail("execution sanity requires valid --draft-model and --target-model")
        else:
            command = [
                sys.executable, str(Path(__file__).with_name("batch_experiments.py")),
                "--prompt-file", str(prompt_path), "--prompt-limit", "1",
                "--methods", "vanilla_dssd,tkslt_as2,confidence_lookahead_dssd,async_sd",
                "--max-len", "8",
                "--draft-model", args.draft_model, "--target-model", args.target_model,
                "--device-draft", args.device_draft, "--device-target", args.device_target,
                "--temperature", str(args.temperature),
                "--sampling-top-k", str(args.sampling_top_k),
                "--sampling-top-p", str(args.sampling_top_p),
                "--reject-topk", str(args.reject_topk),
                "--output", args.output, "--run-tag", "sanity", "--no-resume",
            ]
            print("[RUN ] executing one short prompt")
            result = subprocess.run(command, check=False)
            if result.returncode:
                errors += fail(f"execution sanity exited with code {result.returncode}")
            else:
                ok(f"execution sanity completed: {args.output}")
                errors += validate_csv(args.output)

    if errors:
        print(f"Sanity check failed with {errors} issue(s).")
        return 1
    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
