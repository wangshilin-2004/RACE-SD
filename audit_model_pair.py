#!/usr/bin/env python3
"""Measure draft/target distribution overlap before expensive DSSD runs."""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import dynamic_dssd_main as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-tag", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--device-draft", default="cuda:0")
    parser.add_argument("--device-target", default="cuda:0")
    parser.add_argument("--model-dtype", default="auto")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling-top-k", type=int, default=0)
    parser.add_argument("--sampling-top-p", type=float, default=0.0)
    parser.add_argument("--per-workload-limit", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    # Accepted so the shared experiment argument list can be reused unchanged.
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--lookahead-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--lookahead-min-gamma", type=int, default=1)
    parser.add_argument("--lookahead-max-gamma", type=int, default=8)
    parser.add_argument("--tkslt-k", type=int, default=320)
    parser.add_argument("--tkslt-gamma-set", default="1,2,3,4,6,8")
    parser.add_argument("--tkslt-beta", type=float, default=0.2)
    parser.add_argument("--async-shadow-launch-mode", default="adaptive")
    parser.add_argument("--async-shadow-anchor-mode", default="sample")
    parser.add_argument("--async-shadow-overlap-margin", type=float, default=1.0)
    parser.add_argument("--dynamic-gamma-set", default="1,2,3,4,6,8")
    parser.add_argument("--dynamic-min-gamma", type=int, default=1)
    parser.add_argument("--safe-gamma", type=int, default=1)
    return parser.parse_args()


def load_prompts(path, per_workload_limit):
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
            if index >= per_workload_limit:
                continue
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


def check_tokenizers(draft_tokenizer, target_tokenizer):
    if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise RuntimeError("draft and target token-to-ID mappings differ")
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(draft_tokenizer, attr, None) != getattr(target_tokenizer, attr, None):
            raise RuntimeError(f"draft and target {attr} values differ")


def append_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    check_tokenizers(draft_tokenizer, target_tokenizer)

    dtype = model_dtype(args.model_dtype)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=dtype
    ).to(args.device_draft).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype
    ).to(args.device_target).eval()
    if int(draft_model.config.vocab_size) != int(target_model.config.vocab_size):
        raise RuntimeError("draft and target vocabulary sizes differ")

    prompts = load_prompts(args.prompt_file, args.per_workload_limit)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    rows = []
    progress = tqdm(prompts, desc=f"audit {args.pair_tag}", unit="prompt")
    with torch.no_grad():
        for workload, prompt_id, text in progress:
            prefix = draft_tokenizer.encode(text, return_tensors="pt")
            for step in range(args.steps):
                q_logits = draft_model(prefix.to(args.device_draft)).logits[:, -1, :]
                p_logits = target_model(prefix.to(args.device_target)).logits[:, -1, :]
                q = core.sampling_probs(
                    q_logits, args.temperature, args.sampling_top_k, args.sampling_top_p
                )[0].float().cpu()
                p = core.sampling_probs(
                    p_logits, args.temperature, args.sampling_top_k, args.sampling_top_p
                )[0].float().cpu()

                overlap = float(torch.minimum(p, q).sum().item())
                tv = float(0.5 * torch.abs(p - q).sum().item())
                p_top = int(torch.argmax(p).item())
                q_top = int(torch.argmax(q).item())
                entropy_p = float((-(p * p.clamp_min(1e-12).log())).sum().item())
                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "pair_tag": args.pair_tag,
                    "draft_model": args.draft_model,
                    "target_model": args.target_model,
                    "workload": workload,
                    "prompt_id": prompt_id,
                    "step": step,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "sampling_top_k": args.sampling_top_k,
                    "sampling_top_p": args.sampling_top_p,
                    "theoretical_acceptance": overlap,
                    "tv_distance": tv,
                    "top1_agreement": int(p_top == q_top),
                    "target_prob_at_draft_top1": float(p[q_top].item()),
                    "draft_prob_at_target_top1": float(q[p_top].item()),
                    "target_entropy": entropy_p,
                })

                next_token = torch.multinomial(p, 1, generator=generator).view(1, 1)
                prefix = torch.cat((prefix, next_token), dim=1)

    if not rows:
        raise RuntimeError("no prompts were selected")
    if os.path.exists(args.output):
        os.remove(args.output)
    append_rows(args.output, rows)
    mean_acceptance = sum(row["theoretical_acceptance"] for row in rows) / len(rows)
    mean_tv = sum(row["tv_distance"] for row in rows) / len(rows)
    top1 = sum(row["top1_agreement"] for row in rows) / len(rows)
    print(
        f"{args.pair_tag}: samples={len(rows)} theoretical_acceptance={mean_acceptance:.3f} "
        f"tv={mean_tv:.3f} top1_agreement={top1:.3f} output={args.output}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
