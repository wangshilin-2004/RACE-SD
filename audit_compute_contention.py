#!/usr/bin/env python3
"""Measure whether CPU shadow drafting slows GPU verification on one host."""

import argparse
import csv
import json
import os
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--device-draft", default="cpu")
    parser.add_argument("--device-target", default="cuda:0")
    parser.add_argument("--model-dtype", default="auto")
    parser.add_argument("--per-workload-limit", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shadow-gamma", type=int, default=1)
    return parser.parse_args()


def load_prompts(path, limit):
    counts = defaultdict(int)
    prompts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            workload = str(item.get("source", "unknown"))
            index = counts[workload]
            counts[workload] += 1
            if index < limit:
                prompts.append((workload, f"{workload}-{index:03d}", str(item["text"])))
    return prompts


def model_dtype(name):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start


def main():
    args = parse_args()
    if args.shadow_gamma < 1 or args.repeats < 1:
        raise SystemExit("--shadow-gamma and --repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise RuntimeError("draft and target token-to-ID mappings differ")

    dtype = model_dtype(args.model_dtype)
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=dtype
    ).to(args.device_draft).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype
    ).to(args.device_target).eval()
    rows = []

    with torch.no_grad(), ThreadPoolExecutor(max_workers=2) as executor:
        for workload, prompt_id, text in load_prompts(
            args.prompt_file, args.per_workload_limit
        ):
            prefix = tokenizer.encode(text, return_tensors="pt")
            draft_prefix = prefix.to(args.device_draft)
            first = draft(draft_prefix).logits[:, -1, :]
            token = torch.argmax(first, dim=-1, keepdim=True)
            x_draft = torch.cat((draft_prefix, token), dim=1)

            @torch.inference_mode()
            def shadow_work():
                x = x_draft
                for _ in range(args.shadow_gamma + 1):
                    logits = draft(x).logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    x = torch.cat((x, next_token), dim=1)
                return int(x[0, -1].item())

            @torch.inference_mode()
            def target_work():
                logits = target(x_draft.to(args.device_target)).logits
                return float(logits[0, -1, 0].item())

            # Warm both paths before collecting samples for this prompt length.
            shadow_work()
            target_work()

            for repeat in range(args.repeats):
                if repeat % 2 == 0:
                    shadow_alone = timed(shadow_work, args.device_draft)
                    target_alone = timed(target_work, args.device_target)
                else:
                    target_alone = timed(target_work, args.device_target)
                    shadow_alone = timed(shadow_work, args.device_draft)

                barrier = threading.Barrier(2)

                def concurrent_shadow():
                    barrier.wait()
                    return timed(shadow_work, args.device_draft)

                def concurrent_target():
                    barrier.wait()
                    return timed(target_work, args.device_target)

                wall_start = time.perf_counter()
                shadow_future = executor.submit(concurrent_shadow)
                target_future = executor.submit(concurrent_target)
                shadow_concurrent = shadow_future.result()
                target_concurrent = target_future.result()
                concurrent_wall = time.perf_counter() - wall_start

                rows.append({
                    "workload": workload,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "prompt_tokens": int(prefix.shape[1]),
                    "shadow_gamma": args.shadow_gamma,
                    "draft_model": args.draft_model,
                    "target_model": args.target_model,
                    "device_draft": args.device_draft,
                    "device_target": args.device_target,
                    "shadow_alone_ms": shadow_alone * 1000.0,
                    "target_alone_ms": target_alone * 1000.0,
                    "sequential_ms": (shadow_alone + target_alone) * 1000.0,
                    "ideal_overlap_ms": max(shadow_alone, target_alone) * 1000.0,
                    "shadow_concurrent_ms": shadow_concurrent * 1000.0,
                    "target_concurrent_ms": target_concurrent * 1000.0,
                    "concurrent_wall_ms": concurrent_wall * 1000.0,
                    "shadow_slowdown_ratio": shadow_concurrent / max(shadow_alone, 1e-12),
                    "target_slowdown_ratio": target_concurrent / max(target_alone, 1e-12),
                    "overlap_efficiency": max(shadow_alone, target_alone) / max(concurrent_wall, 1e-12),
                })

    if not rows:
        raise RuntimeError("no prompts selected")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key):
        return statistics.mean(float(row[key]) for row in rows)

    print(f"compute contention audit: samples={len(rows)} output={args.output}")
    print(
        f"  standalone shadow={mean('shadow_alone_ms'):.2f} ms, "
        f"target={mean('target_alone_ms'):.2f} ms, "
        f"ideal-overlap={mean('ideal_overlap_ms'):.2f} ms"
    )
    print(
        f"  concurrent wall={mean('concurrent_wall_ms'):.2f} ms, "
        f"shadow slowdown={mean('shadow_slowdown_ratio'):.3f}x, "
        f"target slowdown={mean('target_slowdown_ratio'):.3f}x, "
        f"overlap efficiency={mean('overlap_efficiency'):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
