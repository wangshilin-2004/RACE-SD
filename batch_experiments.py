#!/usr/bin/env python3
"""Batch runner for Async-SD experiments without reloading models per prompt."""

import argparse
import copy
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import dynamic_dssd_main as core
import external_baselines as baselines


METHODS = (
    "ar_target",
    "dsd",
    "vanilla_dssd",
    "optimized_dssd",
    "tkslt_as2",
    "confidence_lookahead_dssd",
    "pipesd",
    "dynamic_dssd",
    "async_sd",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Async-SD experiments over a JSONL prompt set."
    )
    parser.add_argument("--prompt-file", default="prompts_300.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default="async_sd", help="Comma-separated method names")
    parser.add_argument(
        "--method-order", choices=("fixed", "rotate"), default="rotate",
        help="Rotate method order across prompts to balance execution-order bias",
    )
    parser.add_argument("--workloads", default="", help="Comma-separated source names; empty means all")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int, default=0, help="0 means all selected prompts")
    parser.add_argument("--per-workload-limit", type=int, default=0,
                        help="Take up to N prompts from each selected workload")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--device-draft", default="cpu")
    parser.add_argument("--device-target", default="cuda:0")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--max-len", type=int, default=80)
    dtype_choices = ("auto", "float16", "bfloat16", "float32")
    parser.add_argument("--model-dtype", choices=dtype_choices, default="auto",
                        help="Backward-compatible default for both models")
    parser.add_argument("--draft-model-dtype", choices=dtype_choices, default=None)
    parser.add_argument("--target-model-dtype", choices=dtype_choices, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--cpu-interop-threads", type=int, default=None)
    parser.add_argument("--draft-quantization", choices=("none", "dynamic-int8"), default="none")
    parser.add_argument("--draft-quantized-engine", default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling-top-k", type=int, default=0)
    parser.add_argument("--sampling-top-p", type=float, default=0.0)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--bandwidth", type=float, default=100.0)
    parser.add_argument("--rtt-ms", type=float, default=20.0)
    parser.add_argument("--link-trace-path", default="")
    parser.add_argument("--link-trace-speedup", type=float, default=1.0)
    parser.add_argument(
        "--controller-link-mode",
        choices=("instantaneous", "causal_previous_round"),
        default="causal_previous_round",
        help="Link telemetry exposed to the online controller",
    )
    parser.add_argument("--reject-topk", type=int, default=64)
    parser.add_argument("--mass-threshold", type=float, default=0.995)
    parser.add_argument("--reject-packet-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--reject-min-prob", type=float, default=0.0)
    parser.add_argument("--dynamic-gamma-set", default="1,2,3,4,6,8")
    parser.add_argument("--dynamic-min-gamma", type=int, default=1)
    parser.add_argument("--dynamic-reject-topk-set", default="16,32,64,128")
    parser.add_argument("--dynamic-reject-mass-set", default="0.98,0.99,0.995")
    parser.add_argument("--lookahead-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--lookahead-min-gamma", type=int, default=1)
    parser.add_argument("--lookahead-max-gamma", type=int, default=8)
    parser.add_argument("--tkslt-k", type=int, default=320)
    parser.add_argument("--tkslt-gamma-set", default="1,2,3,4,6,8")
    parser.add_argument("--tkslt-beta", type=float, default=0.2)
    parser.add_argument("--tkslt-warmup-rounds", type=int, default=3)
    parser.add_argument("--tkslt-initial-alpha", type=float, default=0.5)
    parser.add_argument("--tkslt-initial-slm-ms", type=float, default=5.0)
    parser.add_argument("--tkslt-initial-llm-ms", type=float, default=50.0)
    parser.add_argument(
        "--tkslt-packet-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--pipesd-single-threshold", type=float, default=0.94)
    parser.add_argument("--pipesd-multi-threshold", type=float, default=0.90)
    parser.add_argument("--pipesd-min-gamma", type=int, default=1)
    parser.add_argument("--pipesd-max-gamma", type=int, default=8)
    parser.add_argument(
        "--pipesd-packet-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--controller-score", choices=("ratio", "difference"), default="ratio")
    parser.add_argument("--compression-estimator", choices=("legacy", "history"), default="legacy")
    parser.add_argument("--acceptance-predictor", choices=("legacy", "lcb"), default="legacy")
    parser.add_argument("--controller-matrix", action="store_true",
                        help="Interleave five revision variants per prompt; dynamic methods only")
    parser.add_argument("--z-alpha", type=float, default=0.7)
    parser.add_argument("--alpha-weight-history", type=float, default=0.65)
    parser.add_argument("--alpha-weight-difficulty", type=float, default=0.20)
    parser.add_argument("--alpha-weight-early-reject", type=float, default=0.15)
    parser.add_argument("--cmp-weight-mass", type=float, default=0.45)
    parser.add_argument("--cmp-weight-saving", type=float, default=0.25)
    parser.add_argument("--cmp-weight-topk", type=float, default=0.20)
    parser.add_argument("--cmp-weight-easiness", type=float, default=0.10)
    parser.add_argument("--cmp-threshold-penalty", type=float, default=0.15)
    parser.add_argument("--cmp-k-ref", type=float, default=64.0)
    parser.add_argument("--cmp-tau0", type=float, default=0.98)
    parser.add_argument("--cmp-delta-tau", type=float, default=0.02)
    parser.add_argument("--lambda-sw", type=float, default=0.02)
    parser.add_argument("--lambda-cmp", type=float, default=0.02)
    parser.add_argument("--lambda-tail", type=float, default=0.04)
    parser.add_argument("--hysteresis", type=float, default=0.01)
    parser.add_argument("--min-residency", type=int, default=2)
    parser.add_argument("--alpha-panic", type=float, default=0.18)
    parser.add_argument("--safe-gamma", type=int, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=6)
    parser.add_argument("--panic-reject-rate", type=float, default=0.80)
    parser.add_argument("--panic-early-reject", type=float, default=0.70)
    parser.add_argument("--async-shadow-gamma-cap", type=int, default=8)
    parser.add_argument(
        "--async-shadow-launch-mode", choices=("adaptive", "always"), default="adaptive"
    )
    parser.add_argument(
        "--async-shadow-anchor-mode", choices=("sample", "map"), default="sample"
    )
    parser.add_argument("--async-shadow-overlap-margin", type=float, default=1.0)
    parser.add_argument(
        "--controller-variant-file", default="",
        help="JSON list of named controller-parameter overrides, evaluated in one process",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--verbose-method-output", action="store_true",
        help="Show generated text, per-method statistics, and inner progress bars",
    )
    return parser.parse_args()


def validate_args(args):
    for name in ("cpu_threads", "cpu_interop_threads"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "draft_quantization", "none") != "none":
        if args.device_draft != "cpu":
            raise SystemExit("Dynamic INT8 draft quantization requires --device-draft cpu")
        if (args.draft_model_dtype or args.model_dtype) != "float32":
            raise SystemExit("Dynamic INT8 requires --draft-model-dtype float32")
    if args.max_len < 1:
        raise SystemExit("--max-len must be positive")
    if args.gamma < 1:
        raise SystemExit("--gamma must be positive")
    if args.dynamic_min_gamma < 1:
        raise SystemExit("--dynamic-min-gamma must be positive")
    if args.safe_gamma < args.dynamic_min_gamma:
        raise SystemExit("--safe-gamma must be >= --dynamic-min-gamma")
    try:
        dynamic_gammas = {
            int(value.strip()) for value in args.dynamic_gamma_set.split(",")
            if value.strip()
        }
    except ValueError as exc:
        raise SystemExit("--dynamic-gamma-set must be a comma-separated integer list") from exc
    if not dynamic_gammas or not any(
        gamma >= args.dynamic_min_gamma for gamma in dynamic_gammas
    ):
        raise SystemExit("--dynamic-gamma-set must contain an action >= --dynamic-min-gamma")
    if args.safe_gamma not in dynamic_gammas:
        raise SystemExit("--safe-gamma must be included in --dynamic-gamma-set")
    if args.bandwidth <= 0 or args.rtt_ms < 0:
        raise SystemExit("--bandwidth must be positive and --rtt-ms nonnegative")
    if args.link_trace_speedup <= 0:
        raise SystemExit("--link-trace-speedup must be positive")
    if args.sampling_top_k < 0 or not 0.0 <= args.sampling_top_p <= 1.0:
        raise SystemExit("sampling top-k must be nonnegative and top-p must be in [0, 1]")
    if args.reject_topk < 1 or not 0.0 <= args.mass_threshold <= 1.0:
        raise SystemExit("reject top-k must be positive and mass threshold must be in [0, 1]")
    if not 0.0 < args.beta <= 1.0:
        raise SystemExit("--beta must be in (0, 1]")
    nonnegative = {
        "--alpha-weight-history": args.alpha_weight_history,
        "--alpha-weight-difficulty": args.alpha_weight_difficulty,
        "--alpha-weight-early-reject": args.alpha_weight_early_reject,
        "--cmp-weight-mass": args.cmp_weight_mass,
        "--cmp-weight-saving": args.cmp_weight_saving,
        "--cmp-weight-topk": args.cmp_weight_topk,
        "--cmp-weight-easiness": args.cmp_weight_easiness,
        "--cmp-threshold-penalty": args.cmp_threshold_penalty,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0.0]
    if invalid:
        raise SystemExit(f"Controller weights must be nonnegative: {', '.join(invalid)}")
    if sum((args.alpha_weight_history, args.alpha_weight_difficulty,
            args.alpha_weight_early_reject)) <= 0.0:
        raise SystemExit("Acceptance-prediction weights must have positive sum")
    if sum((args.cmp_weight_mass, args.cmp_weight_saving,
            args.cmp_weight_topk, args.cmp_weight_easiness)) <= 0.0:
        raise SystemExit("Compression-proxy weights must have positive sum")
    if args.cmp_k_ref <= 0.0 or args.cmp_delta_tau <= 0.0:
        raise SystemExit("--cmp-k-ref and --cmp-delta-tau must be positive")
    if args.async_shadow_gamma_cap < 1:
        raise SystemExit("--async-shadow-gamma-cap must be positive")
    if args.async_shadow_overlap_margin <= 0.0:
        raise SystemExit("--async-shadow-overlap-margin must be positive")
    if not 0.0 <= args.lookahead_confidence_threshold <= 1.0:
        raise SystemExit("--lookahead-confidence-threshold must be in [0, 1]")
    if args.lookahead_min_gamma < 1 or args.lookahead_max_gamma < args.lookahead_min_gamma:
        raise SystemExit("lookahead gamma bounds must satisfy 1 <= min <= max")
    if args.tkslt_k < 1:
        raise SystemExit("--tkslt-k must be positive")
    if not 0.0 < args.tkslt_beta <= 1.0:
        raise SystemExit("--tkslt-beta must be in (0, 1]")
    if not 0.0 <= args.pipesd_single_threshold <= 1.0:
        raise SystemExit("--pipesd-single-threshold must be in [0, 1]")
    if not 0.0 <= args.pipesd_multi_threshold <= 1.0:
        raise SystemExit("--pipesd-multi-threshold must be in [0, 1]")
    if args.pipesd_min_gamma < 1 or args.pipesd_max_gamma < args.pipesd_min_gamma:
        raise SystemExit("PipeSD gamma bounds must satisfy 1 <= min <= max")


def load_prompts(path, workloads, start, limit, per_workload_limit=0):
    selected = set(filter(None, (x.strip() for x in workloads.split(","))))
    source_indices = {}
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            source = str(item.get("source", "unknown"))
            source_index = source_indices.get(source, 0)
            source_indices[source] = source_index + 1
            if selected and source not in selected:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rows.append({
                "workload": source,
                "prompt_id": f"{source}-{source_index:03d}",
                "line_number": line_number,
                "text": text,
            })
    if per_workload_limit > 0:
        kept = []
        kept_counts = {}
        for row in rows:
            source = row["workload"]
            if kept_counts.get(source, 0) >= per_workload_limit:
                continue
            kept.append(row)
            kept_counts[source] = kept_counts.get(source, 0) + 1
        rows = kept
    rows = rows[max(0, start):]
    return rows[:limit] if limit > 0 else rows


def reset_rng(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_tokenizer_alignment(draft_tokenizer, target_tokenizer):
    """Require exact token-to-ID alignment for distribution-level DSSD."""
    draft_vocab = draft_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    if draft_vocab != target_vocab:
        differing = []
        for token in sorted(set(draft_vocab) | set(target_vocab)):
            if draft_vocab.get(token) != target_vocab.get(token):
                differing.append(
                    f"{token!r}: draft={draft_vocab.get(token)}, target={target_vocab.get(token)}"
                )
            if len(differing) >= 5:
                break
        detail = "; ".join(differing)
        raise RuntimeError(
            "Draft and target token-to-ID mappings differ; DSSD probability alignment "
            f"is invalid. First differences: {detail}"
        )

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(draft_tokenizer, attr, None) != getattr(target_tokenizer, attr, None):
            raise RuntimeError(
                f"Draft and target tokenizer {attr} values differ: "
                f"{getattr(draft_tokenizer, attr, None)} vs "
                f"{getattr(target_tokenizer, attr, None)}"
            )


def config_id(args, method):
    payload = {
        "method": method,
        "method_order": args.method_order,
        "draft": args.draft_model,
        "target": args.target_model,
        "seed": args.seed,
        "max_len": args.max_len,
        "model_dtype": args.model_dtype,
        "draft_model_dtype": args.draft_model_dtype or args.model_dtype,
        "target_model_dtype": args.target_model_dtype or args.model_dtype,
        "cpu_threads": getattr(args, "cpu_threads", None),
        "cpu_interop_threads": getattr(args, "cpu_interop_threads", None),
        "draft_quantization": getattr(args, "draft_quantization", "none"),
        "draft_quantized_engine": getattr(args, "draft_quantized_engine", "auto"),
        "temperature": args.temperature,
        "sampling_top_k": args.sampling_top_k,
        "sampling_top_p": args.sampling_top_p,
        "gamma": args.gamma,
        "bandwidth": args.bandwidth,
        "rtt_ms": args.rtt_ms,
        "trace": args.link_trace_path,
        "trace_sha256": getattr(args, "link_trace_sha256", ""),
        "trace_speedup": args.link_trace_speedup,
        "controller_link_mode": args.controller_link_mode,
        "reject_topk": args.reject_topk,
        "mass_threshold": args.mass_threshold,
        "reject_packet_dtype": args.reject_packet_dtype,
        "reject_min_prob": args.reject_min_prob,
        "dynamic_gamma_set": args.dynamic_gamma_set,
        "dynamic_min_gamma": args.dynamic_min_gamma,
        "dynamic_reject_topk_set": args.dynamic_reject_topk_set,
        "dynamic_reject_mass_set": args.dynamic_reject_mass_set,
        "lookahead_confidence_threshold": args.lookahead_confidence_threshold,
        "lookahead_min_gamma": args.lookahead_min_gamma,
        "lookahead_max_gamma": args.lookahead_max_gamma,
        "tkslt_k": args.tkslt_k,
        "tkslt_gamma_set": args.tkslt_gamma_set,
        "tkslt_beta": args.tkslt_beta,
        "tkslt_warmup_rounds": args.tkslt_warmup_rounds,
        "tkslt_initial_alpha": args.tkslt_initial_alpha,
        "tkslt_initial_slm_ms": args.tkslt_initial_slm_ms,
        "tkslt_initial_llm_ms": args.tkslt_initial_llm_ms,
        "tkslt_packet_dtype": args.tkslt_packet_dtype,
        "pipesd_single_threshold": args.pipesd_single_threshold,
        "pipesd_multi_threshold": args.pipesd_multi_threshold,
        "pipesd_min_gamma": args.pipesd_min_gamma,
        "pipesd_max_gamma": args.pipesd_max_gamma,
        "pipesd_packet_dtype": args.pipesd_packet_dtype,
        "beta": args.beta,
        "controller_score": args.controller_score,
        "compression_estimator": args.compression_estimator,
        "acceptance_predictor": args.acceptance_predictor,
        "z_alpha": args.z_alpha,
        "alpha_weight_history": args.alpha_weight_history,
        "alpha_weight_difficulty": args.alpha_weight_difficulty,
        "alpha_weight_early_reject": args.alpha_weight_early_reject,
        "cmp_weight_mass": args.cmp_weight_mass,
        "cmp_weight_saving": args.cmp_weight_saving,
        "cmp_weight_topk": args.cmp_weight_topk,
        "cmp_weight_easiness": args.cmp_weight_easiness,
        "cmp_threshold_penalty": args.cmp_threshold_penalty,
        "cmp_k_ref": args.cmp_k_ref,
        "cmp_tau0": args.cmp_tau0,
        "cmp_delta_tau": args.cmp_delta_tau,
        "lambda_sw": args.lambda_sw,
        "lambda_cmp": args.lambda_cmp,
        "lambda_tail": args.lambda_tail,
        "hysteresis": args.hysteresis,
        "min_residency": args.min_residency,
        "alpha_panic": args.alpha_panic,
        "safe_gamma": args.safe_gamma,
        "warmup_rounds": args.warmup_rounds,
        "panic_reject_rate": args.panic_reject_rate,
        "panic_early_reject": args.panic_early_reject,
        "async_shadow_gamma_cap": args.async_shadow_gamma_cap,
        "async_shadow_launch_mode": args.async_shadow_launch_mode,
        "async_shadow_anchor_mode": args.async_shadow_anchor_mode,
        "async_shadow_overlap_margin": args.async_shadow_overlap_margin,
        "controller_variant_file_sha256": getattr(
            args, "controller_variant_file_sha256", ""
        ),
        "device_draft": args.device_draft,
        "device_target": args.device_target,
        "run_tag": args.run_tag,
        "prompt_file_sha256": getattr(args, "prompt_file_sha256", ""),
        "code_sha256": getattr(args, "code_sha256", ""),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def completed_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row.get("config_id", ""), row.get("prompt_id", ""), row.get("method", ""))
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def experiment_args(cli):
    args = core.parse_arguments([])
    args.draft_model_name = cli.draft_model
    args.target_model_name = cli.target_model
    args.device_1 = cli.device_draft
    args.device_2 = cli.device_target
    args.device = cli.device_target
    args.seed = cli.seed
    args.max_len = cli.max_len
    args.temperature = cli.temperature
    args.top_k = cli.sampling_top_k
    args.top_p = cli.sampling_top_p
    args.gamma = cli.gamma
    args.bandwidth = cli.bandwidth
    args.rtt = cli.rtt_ms / 1000.0
    args.link_trace_path = cli.link_trace_path
    args.link_trace_speedup = cli.link_trace_speedup
    args.controller_link_mode = cli.controller_link_mode
    args.reject_topk = cli.reject_topk
    args.reject_mass_threshold = cli.mass_threshold
    args.reject_packet_dtype = cli.reject_packet_dtype
    args.reject_min_prob = cli.reject_min_prob
    args.dynamic_gamma_set = cli.dynamic_gamma_set
    args.dynamic_min_gamma = cli.dynamic_min_gamma
    args.dynamic_reject_topk_set = cli.dynamic_reject_topk_set
    args.dynamic_reject_mass_set = cli.dynamic_reject_mass_set
    args.lookahead_confidence_threshold = cli.lookahead_confidence_threshold
    args.lookahead_min_gamma = cli.lookahead_min_gamma
    args.lookahead_max_gamma = cli.lookahead_max_gamma
    args.tkslt_k = cli.tkslt_k
    args.tkslt_gamma_set = cli.tkslt_gamma_set
    args.tkslt_beta = cli.tkslt_beta
    args.tkslt_warmup_rounds = cli.tkslt_warmup_rounds
    args.tkslt_initial_alpha = cli.tkslt_initial_alpha
    args.tkslt_initial_slm_ms = cli.tkslt_initial_slm_ms
    args.tkslt_initial_llm_ms = cli.tkslt_initial_llm_ms
    args.tkslt_packet_dtype = cli.tkslt_packet_dtype
    args.pipesd_single_threshold = cli.pipesd_single_threshold
    args.pipesd_multi_threshold = cli.pipesd_multi_threshold
    args.pipesd_min_gamma = cli.pipesd_min_gamma
    args.pipesd_max_gamma = cli.pipesd_max_gamma
    args.pipesd_packet_dtype = cli.pipesd_packet_dtype
    args.dynamic_beta = cli.beta
    args.dynamic_score_mode = cli.controller_score
    args.dynamic_compression_estimator = cli.compression_estimator
    args.dynamic_acceptance_predictor = cli.acceptance_predictor
    args.dynamic_z_alpha = cli.z_alpha
    args.dynamic_alpha_weight_history = cli.alpha_weight_history
    args.dynamic_alpha_weight_difficulty = cli.alpha_weight_difficulty
    args.dynamic_alpha_weight_early_reject = cli.alpha_weight_early_reject
    args.dynamic_cmp_weight_mass = cli.cmp_weight_mass
    args.dynamic_cmp_weight_saving = cli.cmp_weight_saving
    args.dynamic_cmp_weight_topk = cli.cmp_weight_topk
    args.dynamic_cmp_weight_easiness = cli.cmp_weight_easiness
    args.dynamic_cmp_threshold_penalty = cli.cmp_threshold_penalty
    args.dynamic_cmp_k_ref = cli.cmp_k_ref
    args.dynamic_cmp_tau0 = cli.cmp_tau0
    args.dynamic_cmp_delta_tau = cli.cmp_delta_tau
    args.dynamic_lambda_sw = cli.lambda_sw
    args.dynamic_lambda_cmp = cli.lambda_cmp
    args.dynamic_lambda_tail = cli.lambda_tail
    args.dynamic_hysteresis = cli.hysteresis
    args.dynamic_min_residency = cli.min_residency
    args.dynamic_alpha_panic = cli.alpha_panic
    args.dynamic_safe_gamma = cli.safe_gamma
    args.dynamic_warmup_rounds = cli.warmup_rounds
    args.dynamic_panic_reject_rate = cli.panic_reject_rate
    args.dynamic_panic_early_reject = cli.panic_early_reject
    args.async_shadow_gamma_cap = cli.async_shadow_gamma_cap
    args.dynamic_async_launch_mode = cli.async_shadow_launch_mode
    args.async_shadow_anchor_mode = cli.async_shadow_anchor_mode
    args.async_shadow_overlap_margin = cli.async_shadow_overlap_margin
    args.enable_dynamic_dssd = True
    args.dynamic_joint_packet_control = True
    args.async_shadow_enabled = True
    args.return_runtime_stats = True
    return args


def empty_metrics():
    return {
        "throughput_tok_s": "",
        "total_time_s": "",
        "acceptance_rate": "",
        "total_comm_s": "",
        "uplink_bytes": "",
        "rounds": "",
        "total_slm_s": "",
        "total_llm_s": "",
        "output_hash": "",
    }


def output_hash(output):
    if torch.is_tensor(output):
        payload = json.dumps(output.detach().cpu().reshape(-1).tolist()).encode("utf-8")
    else:
        payload = str(output).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def run_method(method, shared_args, input_ids, tokenizer, draft_model, target_model):
    args = copy.deepcopy(shared_args)
    reset_rng(args.seed)
    device_draft = torch.device(args.device_1)
    device_target = torch.device(args.device_2)
    metrics = empty_metrics()
    stats = {}

    if method == "ar_target":
        generated, total_time, throughput = core.normal_generate(
            target_model, tokenizer, input_ids, device_target, args
        )
        metrics.update(
            throughput_tok_s=throughput,
            total_time_s=total_time,
            output_hash=output_hash(generated),
        )
        return metrics, stats

    uav = core.UAVNode(draft_model, device_draft, args)
    bs = core.BSNode(target_model, device_target, args)
    if method == "dsd":
        result = core.generate_DSD(uav, bs, input_ids, tokenizer, args)
        values, stats = result[:9], result[11]
    elif method in ("vanilla_dssd", "optimized_dssd"):
        args.compress_reject_packet = method == "optimized_dssd"
        uav.args.compress_reject_packet = args.compress_reject_packet
        bs.args.compress_reject_packet = args.compress_reject_packet
        result = core.generate_DSSD(
            uav, bs, input_ids, tokenizer, args,
            run_tag=method.replace("_", " ").title(),
            show_compression_stats=method == "optimized_dssd",
        )
        values, stats = result[:9], result[9]
    elif method == "dynamic_dssd":
        args.async_shadow_enabled = False
        result = core.generate_dynamic_DSSD(
            uav, bs, input_ids, tokenizer, args, run_tag="Dynamic DSSD"
        )
        values, stats = result[:9], result[9]
    elif method == "confidence_lookahead_dssd":
        result = baselines.generate_confidence_lookahead_dssd(
            uav, bs, input_ids, tokenizer, args
        )
        values, stats = result[:9], result[9]
    elif method == "tkslt_as2":
        result = baselines.generate_tkslt_as2(
            uav, bs, input_ids, tokenizer, args
        )
        values, stats = result[:9], result[9]
    elif method == "pipesd":
        result = baselines.generate_pipesd(
            uav, bs, input_ids, tokenizer, args
        )
        values, stats = result[:9], result[9]
    elif method == "async_sd":
        result = core.generate_async_dynamic_DSSD(
            uav, bs, input_ids, tokenizer, args, run_tag="Async-SD"
        )
        values, stats = result[:9], result[9]
    else:
        raise ValueError(f"Unknown method: {method}")

    generated, throughput, total_time, acceptance, comm, uplink, rounds, slm, llm = values
    metrics.update(
        throughput_tok_s=throughput,
        total_time_s=total_time,
        acceptance_rate=acceptance,
        total_comm_s=comm,
        uplink_bytes=uplink,
        rounds=rounds,
        total_slm_s=slm,
        total_llm_s=llm,
        output_hash=output_hash(generated),
    )
    return metrics, stats


def make_row(cli, prompt, method, metrics, stats, status="ok", error=""):
    same_device = cli.device_draft == cli.device_target
    if same_device and cli.device_draft.startswith("cuda"):
        hardware_mode = "single_gpu_emulation"
    elif {cli.device_draft.split(":")[0], cli.device_target.split(":")[0]} == {"cpu", "cuda"}:
        hardware_mode = "cpu_gpu_split"
    elif cli.device_draft.startswith("cuda") and cli.device_target.startswith("cuda"):
        hardware_mode = "separate_cuda_devices"
    else:
        hardware_mode = "heterogeneous_devices"
    def cuda_name(device_name):
        if torch.cuda.is_available() and device_name.startswith("cuda"):
            return torch.cuda.get_device_name(torch.device(device_name))
        return ""

    cuda_draft_name = cuda_name(cli.device_draft)
    cuda_target_name = cuda_name(cli.device_target)
    row = {
        "status": status,
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_tag": cli.run_tag,
        "prompt_file": cli.prompt_file,
        "prompt_file_sha256": getattr(cli, "prompt_file_sha256", ""),
        "code_sha256": getattr(cli, "code_sha256", ""),
        "config_id": config_id(cli, method),
        "method": method,
        "method_order_mode": cli.method_order,
        "draft_model": cli.draft_model,
        "target_model": cli.target_model,
        "device_draft": cli.device_draft,
        "device_target": cli.device_target,
        "hardware_mode": hardware_mode,
        "cuda_device_name": cuda_target_name or cuda_draft_name,
        "cuda_draft_device_name": cuda_draft_name,
        "cuda_target_device_name": cuda_target_name,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "workload": prompt["workload"],
        "prompt_id": prompt["prompt_id"],
        "prompt_line": prompt["line_number"],
        "execution_order_index": getattr(cli, "execution_order_index", ""),
        "input_hash": hashlib.sha1(prompt["text"].encode("utf-8")).hexdigest()[:12],
        "seed": cli.seed,
        "max_len": cli.max_len,
        "model_dtype": cli.model_dtype,
        "draft_model_dtype": cli.draft_model_dtype or cli.model_dtype,
        "target_model_dtype": cli.target_model_dtype or cli.model_dtype,
        "resolved_draft_dtype": getattr(cli, "resolved_draft_dtype", ""),
        "resolved_target_dtype": getattr(cli, "resolved_target_dtype", ""),
        "draft_model_parameter_bytes": getattr(cli, "draft_model_parameter_bytes", ""),
        "target_model_parameter_bytes": getattr(cli, "target_model_parameter_bytes", ""),
        "draft_quantization": getattr(cli, "draft_quantization", "none"),
        "resolved_quantized_engine": getattr(cli, "resolved_quantized_engine", ""),
        "quantized_linear_count": getattr(cli, "quantized_linear_count", 0),
        "draft_state_tensor_bytes": getattr(cli, "draft_state_tensor_bytes", ""),
        "temperature": cli.temperature,
        "sampling_top_k": cli.sampling_top_k,
        "sampling_top_p": cli.sampling_top_p,
        "gamma": cli.gamma,
        "dynamic_gamma_set": cli.dynamic_gamma_set,
        "dynamic_min_gamma": cli.dynamic_min_gamma,
        "dynamic_reject_topk_set": cli.dynamic_reject_topk_set,
        "dynamic_reject_mass_set": cli.dynamic_reject_mass_set,
        "lookahead_confidence_threshold": cli.lookahead_confidence_threshold,
        "lookahead_min_gamma": cli.lookahead_min_gamma,
        "lookahead_max_gamma": cli.lookahead_max_gamma,
        "tkslt_k": cli.tkslt_k,
        "tkslt_gamma_set": cli.tkslt_gamma_set,
        "tkslt_beta": cli.tkslt_beta,
        "tkslt_warmup_rounds": cli.tkslt_warmup_rounds,
        "tkslt_initial_alpha": cli.tkslt_initial_alpha,
        "tkslt_initial_slm_ms": cli.tkslt_initial_slm_ms,
        "tkslt_initial_llm_ms": cli.tkslt_initial_llm_ms,
        "tkslt_packet_dtype": cli.tkslt_packet_dtype,
        "pipesd_single_threshold": cli.pipesd_single_threshold,
        "pipesd_multi_threshold": cli.pipesd_multi_threshold,
        "pipesd_min_gamma": cli.pipesd_min_gamma,
        "pipesd_max_gamma": cli.pipesd_max_gamma,
        "pipesd_packet_dtype": cli.pipesd_packet_dtype,
        "beta": cli.beta,
        "controller_score": cli.controller_score,
        "compression_estimator": cli.compression_estimator,
        "acceptance_predictor": cli.acceptance_predictor,
        "z_alpha": cli.z_alpha,
        "alpha_weight_history": cli.alpha_weight_history,
        "alpha_weight_difficulty": cli.alpha_weight_difficulty,
        "alpha_weight_early_reject": cli.alpha_weight_early_reject,
        "cmp_weight_mass": cli.cmp_weight_mass,
        "cmp_weight_saving": cli.cmp_weight_saving,
        "cmp_weight_topk": cli.cmp_weight_topk,
        "cmp_weight_easiness": cli.cmp_weight_easiness,
        "cmp_threshold_penalty": cli.cmp_threshold_penalty,
        "cmp_k_ref": cli.cmp_k_ref,
        "cmp_tau0": cli.cmp_tau0,
        "cmp_delta_tau": cli.cmp_delta_tau,
        "lambda_sw": cli.lambda_sw,
        "lambda_cmp": cli.lambda_cmp,
        "lambda_tail": cli.lambda_tail,
        "hysteresis": cli.hysteresis,
        "min_residency": cli.min_residency,
        "alpha_panic": cli.alpha_panic,
        "safe_gamma": cli.safe_gamma,
        "warmup_rounds": cli.warmup_rounds,
        "panic_reject_rate": cli.panic_reject_rate,
        "panic_early_reject": cli.panic_early_reject,
        "bandwidth_mbps": cli.bandwidth,
        "rtt_ms": cli.rtt_ms,
        "link_trace_path": cli.link_trace_path,
        "link_trace_sha256": getattr(cli, "link_trace_sha256", ""),
        "link_trace_speedup": cli.link_trace_speedup,
        "controller_link_mode": cli.controller_link_mode,
        "reject_topk": cli.reject_topk,
        "mass_threshold": cli.mass_threshold,
        "reject_packet_dtype": cli.reject_packet_dtype,
        "reject_min_prob": cli.reject_min_prob,
        "async_enabled": int(method in {"async_sd", "pipesd"}),
        "async_shadow_gamma_cap": cli.async_shadow_gamma_cap,
        "async_shadow_launch_mode": cli.async_shadow_launch_mode,
        "async_shadow_anchor_mode": cli.async_shadow_anchor_mode,
        "async_shadow_overlap_margin": cli.async_shadow_overlap_margin,
        "controller_variant_file": cli.controller_variant_file,
        "controller_variant_file_sha256": getattr(
            cli, "controller_variant_file_sha256", ""
        ),
    }
    row.update(metrics)
    if metrics.get("throughput_tok_s", "") != "" and metrics.get("total_time_s", "") != "":
        row["generated_tokens"] = int(round(
            float(metrics["throughput_tok_s"]) * float(metrics["total_time_s"])
        ))
    else:
        row["generated_tokens"] = ""
    gamma_usage = stats.get("gamma_usage", {})
    if gamma_usage:
        proposed_tokens = sum(int(gamma) * int(count) for gamma, count in gamma_usage.items())
    elif metrics.get("rounds", "") != "":
        proposed_tokens = int(metrics["rounds"]) * int(cli.gamma)
    else:
        proposed_tokens = ""
    row["proposed_tokens"] = proposed_tokens
    row["accepted_draft_tokens_est"] = (
        float(metrics["acceptance_rate"]) * proposed_tokens
        if proposed_tokens != "" and metrics.get("acceptance_rate", "") != ""
        else ""
    )
    stat_fields = (
        "accept_all_rounds", "reject_rounds", "reject_compressed_cnt", "reject_full_cnt",
        "compression_hit_rate", "reject_down_bytes_actual", "reject_down_bytes_full_equiv",
        "reject_downlink_saving", "avg_kept_mass", "avg_tail_mass",
        "avg_candidate_kept_mass", "avg_candidate_tail_mass",
        "candidate_mass_samples", "avg_gamma",
        "compression_on_ratio", "controller_switch_count", "controller_switch_rate",
        "controller_safe_mode_count", "controller_safe_mode_rate", "async_shadow_launches",
        "controller_warmup_count", "controller_warmup_rate",
        "controller_hold_count", "controller_hold_rate",
        "controller_alpha_ema_final", "controller_alpha_lcb_final",
        "controller_reject_rate_ema_final", "controller_early_reject_ema_final",
        "controller_throughput_estimate_final", "controller_compression_observations",
        "controller_compression_history_pairs", "controller_compression_brier_mean",
        "preview_time_total_s", "preview_fresh_count", "preview_cached_count",
        "preview_time_per_fresh_round_ms", "controller_decision_time_total_s",
        "controller_decision_time_per_round_ms",
        "controller_link_observation_count", "controller_link_bw_up_mbps_final",
        "controller_link_bw_down_mbps_final", "controller_link_rtt_ms_final",
        "async_shadow_launch_rate", "async_shadow_hits", "async_shadow_hit_rate",
        "async_shadow_misses", "async_shadow_discard_reject", "async_shadow_discard_mismatch",
        "async_shadow_discard_gamma_change", "async_shadow_discard_prefix_mismatch",
        "async_shadow_discard_generation_end",
        "async_shadow_partial_extensions", "async_shadow_eligible_rounds",
        "async_shadow_blocked_overlap", "async_shadow_blocked_benefit",
        "async_shadow_cancelled", "async_shadow_accept_all_launches",
        "async_shadow_anchor_matches", "async_shadow_anchor_match_rate",
        "async_shadow_tokens_computed", "async_shadow_predicted_reuse_mean",
        "async_shadow_expected_benefit_mean_s",
        "async_shadow_total_time_s", "async_shadow_wait_time_s",
        "async_shadow_wasted_compute_time_s", "async_shadow_reused_compute_time_s",
        "async_shadow_saved_draft_time_s", "async_shadow_compute_per_token_ema_s",
        "lookahead_confidence_threshold", "lookahead_min_gamma",
        "lookahead_max_gamma", "tkslt_uplink_sparse_ratio", "tkslt_alpha_final",
        "tkslt_predicted_speedup_mean", "tkslt_as2_fallback_rounds",
        "tkslt_as2_fallback_rate",
        "pipesd_trigger_single_count", "pipesd_trigger_multi_count",
        "pipesd_trigger_both_count", "pipesd_dp_batch_count",
        "pipesd_dp_batches_per_round", "pipesd_uplink_service_time_s",
        "pipesd_uplink_hidden_time_s", "pipesd_waiting_launches",
        "pipesd_waiting_hits", "pipesd_waiting_reuse_rate",
        "pipesd_waiting_discard_reject", "pipesd_waiting_discard_mismatch",
        "pipesd_waiting_tokens", "pipesd_rollback_tokens",
        "pipesd_waiting_compute_time_s", "pipesd_waiting_overhang_time_s",
    )
    for name in stat_fields:
        row[name] = stats.get(name, "")
    accepted_est = row.get("accepted_draft_tokens_est", "")
    reject_rounds = row.get("reject_rounds", "")
    if accepted_est != "" and reject_rounds != "":
        verified = float(accepted_est) + float(reject_rounds)
        row["verified_token_acceptance_rate"] = (
            float(accepted_est) / verified if verified > 0.0 else 0.0
        )
    else:
        row["verified_token_acceptance_rate"] = ""
    rounds = metrics.get("rounds", "")
    accept_all_rounds = row.get("accept_all_rounds", "")
    row["accept_all_round_rate"] = (
        float(accept_all_rounds) / float(rounds)
        if rounds not in ("", 0) and accept_all_rounds != ""
        else ""
    )
    for component in ("total", "comm", "slm", "llm"):
        for percentile in ("mean", "p50", "p95", "p99"):
            key = f"round_{component}_latency_{percentile}_ms"
            row[key] = stats.get(key, "")
        samples_key = f"round_{component}_latency_samples_ms"
        row[samples_key] = json.dumps(stats.get(samples_key, []))
    for name in (
        "gamma_usage", "async_shadow_reused_gamma_usage",
        "async_shadow_launch_gamma_usage",
    ):
        row[name] = json.dumps(stats.get(name, {}), sort_keys=True)
    return row


def append_row(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        with open(path, newline="", encoding="utf-8") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != list(row.keys()):
            raise RuntimeError(
                f"CSV schema mismatch for {path}; use a new output path or remove the stale file"
            )
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def model_dtype(name):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def quantize_cpu_draft(model, cli):
    cli.resolved_quantized_engine = ""
    cli.quantized_linear_count = 0
    if cli.draft_quantization == "none":
        return model
    supported = set(torch.backends.quantized.supported_engines) - {"none"}
    engine = cli.draft_quantized_engine
    if engine == "auto":
        preferences = ("qnnpack",) if platform.machine().lower() in {"aarch64", "arm64"} else ("x86", "fbgemm", "qnnpack")
        engine = next((name for name in preferences if name in supported), "")
    if engine not in supported:
        raise RuntimeError(f"No requested CPU INT8 backend; available engines: {sorted(supported)}")
    torch.backends.quantized.engine = engine
    from torch.ao.quantization import quantize_dynamic
    from torch.ao.nn.quantized.dynamic import Linear as QuantizedLinear
    model = quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
    cli.quantized_linear_count = sum(isinstance(module, QuantizedLinear) for module in model.modules())
    if not cli.quantized_linear_count:
        raise RuntimeError("INT8 conversion produced no quantized Linear layers")
    cli.resolved_quantized_engine = engine
    return model


def state_tensor_bytes(model):
    # Quantized Linear weights live in packed state, not ordinary parameters.
    seen = set()
    def size(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(item) for item in value)
        return 0
    return size(model.state_dict())


def warmup_models(draft_model, target_model, input_ids, cli):
    pairs = (
        (draft_model, torch.device(cli.device_draft)),
        (target_model, torch.device(cli.device_target)),
    )
    for model, device in pairs:
        model.to(device)
        model.eval()
        with torch.no_grad():
            model(input_ids[:, :min(input_ids.shape[1], 32)].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def controller_variants(cli):
    variant_file = getattr(cli, "controller_variant_file", "")
    if variant_file:
        if cli.controller_matrix:
            raise SystemExit("Use either --controller-matrix or --controller-variant-file")
        path = Path(variant_file).expanduser()
        if not path.is_file():
            raise SystemExit(f"Controller variant file does not exist: {path}")
        variants_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(variants_data, list) or not variants_data:
            raise SystemExit("Controller variant file must contain a non-empty JSON list")
        allowed = {
            "z_alpha",
            "alpha_weight_history", "alpha_weight_difficulty",
            "alpha_weight_early_reject",
            "cmp_weight_mass", "cmp_weight_saving", "cmp_weight_topk",
            "cmp_weight_easiness", "cmp_threshold_penalty",
            "cmp_k_ref", "cmp_tau0", "cmp_delta_tau",
            "lambda_sw", "lambda_cmp", "lambda_tail",
        }
        variants = []
        seen_names = set()
        for item in variants_data:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise SystemExit("Each controller variant needs a string name")
            name = item["name"].strip()
            overrides = item.get("overrides", {})
            if not name or name in seen_names or not isinstance(overrides, dict):
                raise SystemExit(f"Invalid or duplicate controller variant: {name!r}")
            unknown = sorted(set(overrides) - allowed)
            if unknown:
                raise SystemExit(
                    f"Unsupported override(s) in {name}: {', '.join(unknown)}"
                )
            config = copy.deepcopy(cli)
            for key, value in overrides.items():
                setattr(config, key, value)
            validate_args(config)
            config.run_tag = f"{cli.run_tag}_{name}" if cli.run_tag else name
            variants.append((name, config))
            seen_names.add(name)
        return variants
    if not cli.controller_matrix:
        return [("configured", cli)]
    variants = []
    for name, score, compression, acceptance in (
        ("legacy", "ratio", "legacy", "legacy"),
        ("difference", "difference", "legacy", "legacy"),
        ("history", "ratio", "history", "legacy"),
        ("combined", "difference", "history", "legacy"),
        ("simplified", "difference", "history", "lcb"),
    ):
        config = copy.deepcopy(cli)
        config.controller_score = score
        config.compression_estimator = compression
        config.acceptance_predictor = acceptance
        config.run_tag = f"{cli.run_tag}_{name}"
        variants.append((name, config))
    return variants


def ordered_runs(variants, methods, prompt_index, order_mode):
    runs = [(name, config, method) for name, config in variants for method in methods]
    if order_mode == "rotate" and runs:
        offset = prompt_index % len(runs)
        runs = runs[offset:] + runs[:offset]
    return runs


def main():
    cli = parse_args()
    validate_args(cli)
    if cli.cpu_threads is not None:
        torch.set_num_threads(cli.cpu_threads)
    if cli.cpu_interop_threads is not None:
        torch.set_num_interop_threads(cli.cpu_interop_threads)
    with open(cli.prompt_file, "rb") as prompt_handle:
        cli.prompt_file_sha256 = hashlib.sha256(prompt_handle.read()).hexdigest()
    cli.link_trace_sha256 = ""
    if cli.link_trace_path:
        trace_path = Path(cli.link_trace_path).expanduser()
        if not trace_path.is_file():
            raise SystemExit(f"Link trace does not exist: {trace_path}")
        cli.link_trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    cli.controller_variant_file_sha256 = ""
    if cli.controller_variant_file:
        variant_path = Path(cli.controller_variant_file).expanduser()
        if not variant_path.is_file():
            raise SystemExit(f"Controller variant file does not exist: {variant_path}")
        cli.controller_variant_file_sha256 = hashlib.sha256(
            variant_path.read_bytes()
        ).hexdigest()
    code_hasher = hashlib.sha256()
    for code_path in (
        Path(__file__),
        Path(__file__).with_name("dynamic_dssd_main.py"),
        Path(__file__).with_name("external_baselines.py"),
        Path(__file__).with_name("pipesd_utils.py"),
        Path(__file__).with_name("speculative.py"),
    ):
        code_hasher.update(code_path.read_bytes())
    cli.code_sha256 = code_hasher.hexdigest()
    methods = [x.strip() for x in cli.methods.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    if (cli.controller_matrix or cli.controller_variant_file) and set(methods) - {
        "dynamic_dssd", "async_sd"
    }:
        raise SystemExit(
            "Controller variants support only dynamic_dssd and async_sd"
        )
    prompts = load_prompts(
        cli.prompt_file, cli.workloads, cli.prompt_start, cli.prompt_limit,
        cli.per_workload_limit,
    )
    if not prompts:
        raise SystemExit("No prompts selected")

    done = set() if cli.no_resume else completed_keys(cli.output)
    tokenizer = AutoTokenizer.from_pretrained(cli.draft_model)
    target_tokenizer = AutoTokenizer.from_pretrained(cli.target_model)
    validate_tokenizer_alignment(tokenizer, target_tokenizer)
    draft_dtype = model_dtype(cli.draft_model_dtype or cli.model_dtype)
    target_dtype = model_dtype(cli.target_model_dtype or cli.model_dtype)
    draft_model = AutoModelForCausalLM.from_pretrained(
        cli.draft_model, torch_dtype=draft_dtype
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        cli.target_model, torch_dtype=target_dtype
    )
    draft_model = quantize_cpu_draft(draft_model, cli)
    cli.draft_state_tensor_bytes = state_tensor_bytes(draft_model)
    cli.resolved_draft_dtype = str(next(draft_model.parameters()).dtype)
    cli.resolved_target_dtype = str(next(target_model.parameters()).dtype)
    cli.draft_model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in draft_model.parameters()
    )
    cli.target_model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in target_model.parameters()
    )
    draft_model.eval()
    target_model.eval()
    if int(draft_model.config.vocab_size) != int(target_model.config.vocab_size):
        raise RuntimeError(
            "Draft and target vocabulary sizes differ; DSSD token/probability alignment is invalid"
        )
    if not cli.skip_warmup:
        warmup_ids = tokenizer.encode(prompts[0]["text"], return_tensors="pt")
        print("Warming up draft and target models once before measured runs")
        warmup_models(draft_model, target_model, warmup_ids, cli)

    variants = controller_variants(cli)
    total = len(prompts) * len(methods) * len(variants)
    progress = tqdm(total=total, desc=cli.run_tag or "experiment", unit="run", dynamic_ncols=True)
    try:
        for prompt_index, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt["text"], return_tensors="pt")
            runs = ordered_runs(variants, methods, prompt_index, cli.method_order)
            for order_index, (variant, run_cli, method) in enumerate(runs):
                run_cli.execution_order_index = order_index
                shared_args = experiment_args(run_cli)
                key = (config_id(run_cli, method), prompt["prompt_id"], method)
                if key in done:
                    progress.set_postfix_str(f"skip {method} {prompt['prompt_id']}", refresh=False)
                    progress.update(1)
                    continue

                progress.set_postfix_str(f"run {variant}/{method} {prompt['prompt_id']}", refresh=True)
                captured = io.StringIO()
                try:
                    if cli.verbose_method_output:
                        metrics, stats = run_method(
                            method, shared_args, input_ids, tokenizer, draft_model, target_model
                        )
                    else:
                        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                            metrics, stats = run_method(
                                method, shared_args, input_ids, tokenizer, draft_model, target_model
                            )
                    row = make_row(run_cli, prompt, method, metrics, stats)
                except Exception as exc:
                    row = make_row(
                        run_cli, prompt, method, empty_metrics(), {}, status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    append_row(cli.output, row)
                    progress.write(f"[ERROR] {method} {prompt['prompt_id']}: {type(exc).__name__}: {exc}")
                    detail_lines = captured.getvalue().strip().splitlines()
                    if detail_lines:
                        progress.write("\n".join(detail_lines[-20:]))
                    raise
                append_row(cli.output, row)
                done.add(key)
                throughput = metrics.get("throughput_tok_s", "")
                suffix = f"{float(throughput):.2f} tok/s" if throughput != "" else "ok"
                progress.set_postfix_str(
                    f"{variant}/{method} {prompt['prompt_id']} | {suffix}", refresh=False
                )
                progress.update(1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        progress.close()


if __name__ == "__main__":
    main()
