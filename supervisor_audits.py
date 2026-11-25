#!/usr/bin/env python3
"""Run isolated supervisor-comment audits; no Torch import is needed to plan."""

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE_FILES = ("batch_experiments.py", "dynamic_dssd_main.py", "external_baselines.py",
              "pipesd_utils.py", "speculative.py")
REQUIRED = ("sanity", "precision", "timing", "sensitivity", "sensitivity_poor", "reference", "action_space")
OPTIONAL = ("gamma_diagnosis", "gamma_confirmation", "bf16_gamma_pilot", "quantization_sanity", "quantization", "baseline_sanity", "baseline_pilot", "baseline_main")


def code_hash():
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "all", "report", "unit") + REQUIRED + OPTIONAL)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results_icc2027_supervisor_r2")
    p.add_argument("--prompt-file", type=Path, default=ROOT / "prompts_300.jsonl")
    p.add_argument("--draft-model", default="models/vicuna-68m")
    p.add_argument("--target-model", default="models/vicuna-7b-v1.5")
    p.add_argument("--device-draft", default="cpu")
    p.add_argument("--device-target", default="cuda:0")
    p.add_argument("--int8-engine", default="auto")
    for name, default in (("seed", 321), ("max-len", 80), ("cpu-threads", 20),
                          ("cpu-interop-threads", 20), ("sensitivity-per-workload", 30),
                          ("confirmation-per-workload", 10), ("precision-per-workload", 10),
                          ("precision-repeats", 2), ("timing-per-workload", 10)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("seconds-good", 10), ("seconds-poor", 15), ("seconds-int8", 15)):
        p.add_argument("--" + name, type=float, default=default)
    return p.parse_args()


def batch(tag, methods="async_sd", per_workload=10, **options):
    return dict(tag=tag, methods=methods, per_workload=per_workload, options=options)


def plan(stage, args):
    n = args.confirmation_per_workload
    if stage == "unit":
        return []
    if stage == "sanity":
        entries = [batch("sanity_oat", per_workload=1, max_len=16,
                         controller_variant_file=str(ROOT / "controller_oat_variants.json")),
                   batch("sanity_reference", per_workload=1, max_len=16,
                         controller_variant_file=str(ROOT / "controller_reference_variants.json"))]
        for entry in plan("action_space", args):
            if "action_space_good" in entry["tag"]:
                entries.append(batch("sanity_" + entry["tag"], per_workload=1,
                                     **dict(entry["options"], max_len=16)))
        for dtype in ("float32", "bfloat16"):
            entries.append(batch("sanity_precision_" + dtype, "dynamic_dssd,async_sd", 1,
                                 max_len=16, draft_model_dtype=dtype))
        return entries
    if stage in {"sensitivity", "sensitivity_poor"}:
        poor = stage.endswith("poor")
        return [batch(stage, per_workload=n if poor else args.sensitivity_per_workload,
                      bandwidth=5 if poor else 100, rtt_ms=100 if poor else 20,
                      controller_variant_file=str(ROOT / "controller_oat_variants.json"))]
    if stage == "reference":
        return [batch(f"reference_{link}", per_workload=n, bandwidth=bw, rtt_ms=rtt,
                      controller_variant_file=str(ROOT / "controller_reference_variants.json"))
                for link, bw, rtt in (("good", 100, 20), ("poor", 5, 100))]
    if stage == "action_space":
        return [batch(f"action_space_{link}_{name}", per_workload=n,
                      bandwidth=bw, rtt_ms=rtt, dynamic_reject_topk_set=ks, dynamic_reject_mass_set=masses)
                for link, bw, rtt in (("good", 100, 20), ("poor", 5, 100))
                for name, ks, masses in (
                    ("baseline", "16,32,64,128", "0.98,0.99,0.995"),
                    ("narrow", "32,64,128", "0.99,0.995"),
                    ("wide", "8,16,32,64,128,256", "0.97,0.98,0.99,0.995,0.999"))]
    if stage in {"precision", "quantization", "quantization_sanity"}:
        sanity = stage.endswith("sanity")
        modes = ("float32", "dynamic-int8") if stage.startswith("quantization") else ("float32", "bfloat16")
        entries = []
        for repeat in range(1, (1 if sanity else args.precision_repeats) + 1):
            for mode in (modes if repeat % 2 else modes[::-1]):
                options = dict(draft_model_dtype="float32" if mode == "dynamic-int8" else mode,
                               draft_quantization="dynamic-int8" if mode == "dynamic-int8" else "none",
                               draft_quantized_engine=args.int8_engine)
                if sanity:
                    options["max_len"] = 16
                methods = "dynamic_dssd,async_sd" if repeat % 2 else "async_sd,dynamic_dssd"
                entries.append(batch(f"{stage}_r{repeat}_{mode}", methods,
                                     1 if sanity else args.precision_per_workload, **options))
        return entries
    if stage == "timing":
        return [batch("timing", "dynamic_dssd,async_sd", args.timing_per_workload)]
    if stage == "bf16_gamma_pilot":
        entries = []
        for entry in plan("gamma_diagnosis", args):
            if entry["methods"] == "ar_target":
                entry["options"]["draft_model_dtype"] = "bfloat16"
                entries.append(entry)
            elif entry["options"].get("draft_model_dtype") == "bfloat16":
                entries.append(entry)
        return entries
    if stage == "gamma_confirmation":
        entries = []
        for repeat in range(1, args.precision_repeats + 1):
            configs = [batch(f"confirm_r{repeat}_fixed_g{gamma}", "optimized_dssd", 100,
                             gamma=gamma, draft_model_dtype="bfloat16") for gamma in (1, 2)]
            configs.append(batch(f"confirm_r{repeat}_adaptive",
                                 "ar_target,dynamic_dssd,async_sd" if repeat % 2
                                 else "async_sd,dynamic_dssd,ar_target", 100,
                                 draft_model_dtype="bfloat16"))
            entries.extend(configs if repeat % 2 else configs[::-1])
        return entries
    if stage == "gamma_diagnosis":
        entries = []
        for repeat in range(1, args.precision_repeats + 1):
            modes = ("float32", "bfloat16") if repeat % 2 else ("bfloat16", "float32")
            gammas = (1, 2, 3, 4, 6, 8) if repeat % 2 else (8, 6, 4, 3, 2, 1)
            reference = batch(f"gamma_r{repeat}_ar", "ar_target", n)
            if repeat % 2:
                entries.append(reference)
            for dtype in modes:
                adaptive = batch(f"gamma_r{repeat}_{dtype}_adaptive",
                                 "dynamic_dssd,async_sd" if repeat % 2 else "async_sd,dynamic_dssd",
                                 n, draft_model_dtype=dtype)
                if repeat % 2 == 0:
                    entries.append(adaptive)
                for gamma in gammas:
                    entries.append(batch(f"gamma_r{repeat}_{dtype}_fixed_g{gamma}",
                                         "optimized_dssd", n, gamma=gamma, draft_model_dtype=dtype))
                if repeat % 2:
                    entries.append(adaptive)
            if repeat % 2 == 0:
                entries.append(reference)
        return entries
    if stage == "baseline_sanity":
        return [batch(stage, "pipesd,dynamic_dssd,async_sd", 1, max_len=16)]
    if stage == "baseline_pilot":
        return [batch(f"baseline_pilot_s{s}_m{m}", "pipesd", 10,
                      pipesd_single_threshold=s, pipesd_multi_threshold=m)
                for s, m in ((0.90, 0.85), (0.94, 0.90), (0.97, 0.90), (0.94, 0.95))]
    if stage == "baseline_main":
        path = args.output_dir / "baseline_selected_thresholds.json"
        if not path.is_file():
            raise ValueError("Run baseline_pilot before baseline_main; thresholds are not selected")
        selected = json.loads(path.read_text())
        if selected["code_sha256"] != code_hash() or selected["pilot_sha256"] != file_hash(args.output_dir / "baseline_pilot.csv"):
            raise ValueError("Baseline pilot provenance changed; use a fresh audit directory")
        return [batch(stage, "pipesd,dynamic_dssd,async_sd", 100,
                      pipesd_single_threshold=selected["single"], pipesd_multi_threshold=selected["multi"])]
    raise ValueError(f"Unsupported stage: {stage}")


def configurations(entries):
    result = []
    for entry in entries:
        path = entry["options"].get("controller_variant_file")
        variants = json.loads(Path(path).read_text()) if path else [{"name": None, "overrides": {}}]
        for variant in variants:
            tag = entry["tag"] + ("_" + variant["name"] if variant["name"] else "")
            for method in entry["methods"].split(","):
                result.append((tag, method, entry, variant["overrides"]))
    return result


def print_plan(stage, entries, args):
    runs, seconds = 0, 0.0
    for _, _, entry, _ in configurations(entries):
        count = 3 * entry["per_workload"]
        options = entry["options"]
        duration = args.seconds_poor if options.get("bandwidth") == 5 else args.seconds_good
        if options.get("draft_quantization") == "dynamic-int8":
            duration = args.seconds_int8
        runs += count
        seconds += count * duration * options.get("max_len", args.max_len) / args.max_len
    print(f"{stage:22s} {runs:5d} runs | generation estimate {seconds / 60:.1f} min ({seconds / 3600:.2f} h)", flush=True)
    return runs


def common_options(args):
    return dict(prompt_file=str(args.prompt_file.resolve()), draft_model=args.draft_model,
                target_model=args.target_model, device_draft=args.device_draft, device_target=args.device_target,
                seed=args.seed, max_len=args.max_len, model_dtype="auto", draft_model_dtype="float32",
                target_model_dtype="float16", cpu_threads=args.cpu_threads,
                cpu_interop_threads=args.cpu_interop_threads, temperature=0.7,
                sampling_top_k=0, sampling_top_p=0.0, method_order="rotate", bandwidth=100, rtt_ms=20,
                controller_score="ratio", compression_estimator="legacy", acceptance_predictor="legacy")


def command(entry, args, path):
    options = dict(common_options(args), **entry["options"])
    options.update(methods=entry["methods"], per_workload_limit=entry["per_workload"],
                   run_tag=entry["tag"], output=str(path))
    result = [sys.executable, str(ROOT / "batch_experiments.py")]
    for key, value in options.items():
        result.extend(("--" + key.replace("_", "-"), str(value)))
    return result


def read_latest(path):
    latest = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            latest[(row["config_id"], row["prompt_id"], row["method"])] = row
    return list(latest.values())


def check_stage(path, entries, args):
    rows = read_latest(path)
    expected = {(tag, method): (entry, overrides) for tag, method, entry, overrides in configurations(entries)}
    groups = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            raise ValueError(f"Unresolved error: {row['prompt_id']} {row.get('error')}")
        if row["code_sha256"] != code_hash():
            raise ValueError("CSV code fingerprint differs from this checkout")
        groups[(row["run_tag"], row["method"])].append(row)
    if set(groups) != set(expected):
        raise ValueError(f"Coverage mismatch: missing={set(expected)-set(groups)}, extra={set(groups)-set(expected)}")
    prompt_sets = defaultdict(list)
    for key, group in groups.items():
        entry, overrides = expected[key]
        n = 3 * entry["per_workload"]
        identities = {(r["prompt_id"], r["input_hash"]) for r in group}
        if len(group) != n or len(identities) != n:
            raise ValueError(f"{key}: expected {n} distinct prompts, found {len(group)} rows")
        prompt_sets[n].append(identities)
        options = dict(common_options(args), **entry["options"])
        options.update(overrides)
        for row in group:
            for field, value in options.items():
                if field in {"prompt_file", "controller_variant_file"}:
                    hash_field = "prompt_file_sha256" if field == "prompt_file" else "controller_variant_file_sha256"
                    if row[hash_field] != file_hash(value):
                        raise ValueError(f"{field} fingerprint changed")
                    continue
                csv_field = {"bandwidth": "bandwidth_mbps", "cpu_threads": "torch_num_threads",
                             "cpu_interop_threads": "torch_num_interop_threads", "method_order": "method_order_mode"}.get(field, field)
                # The requested engine can be auto; the resolved backend is checked separately.
                if field == "draft_quantized_engine":
                    continue
                actual = row.get(csv_field, "none" if field == "draft_quantization" else "")
                matches = (actual != "" and math.isclose(float(actual), value, abs_tol=1e-6)
                           if isinstance(value, (int, float)) else actual == str(value))
                if not matches:
                    raise ValueError(f"{key}: {csv_field} expected {value}, found {actual}")
            if row["resolved_draft_dtype"] != "torch." + options["draft_model_dtype"] or row["resolved_target_dtype"] != "torch.float16":
                raise ValueError(f"{key}: resolved model precision differs from request")
            if options.get("draft_quantization") == "dynamic-int8":
                if int(row.get("quantized_linear_count", 0)) <= 0 or not row.get("resolved_quantized_engine"):
                    raise ValueError("INT8 conversion was not verified")
                engine = options.get("draft_quantized_engine", "auto")
                if engine != "auto" and row["resolved_quantized_engine"] != engine:
                    raise ValueError("INT8 backend differs from the requested backend")
    for sets in prompt_sets.values():
        if any(items != sets[0] for items in sets):
            raise ValueError("Compared configurations do not use identical prompts")
    print(f"[OK] {len(rows)} rows: complete configurations, paired prompts, precision, threads, and provenance", flush=True)
    return rows


def run_command(cmd, logfile):
    with logfile.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + json.dumps(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
        if status:
            raise subprocess.CalledProcessError(status, cmd)


def run_unit(args):
    for name in ("test_shadow_pipeline.py", "test_controller_revision.py", "test_supervisor_audits.py",
                 "test_residual_sanity.py", "test_pipesd_baseline.py"):
        run_command([sys.executable, str(ROOT / name)], args.output_dir / "unit.log")


def run_stage(stage, args):
    if stage == "unit":
        run_unit(args)
        return
    if stage == "sanity":
        run_unit(args)
    entries = plan(stage, args)
    print_plan(stage, entries, args)
    path = args.output_dir / f"{stage}.csv"
    manifest = dict(code_sha256=code_hash(), prompts_sha256=file_hash(args.prompt_file),
                    common=common_options(args), entries=entries,
                    variants={e["options"]["controller_variant_file"]: file_hash(e["options"]["controller_variant_file"])
                              for e in entries if "controller_variant_file" in e["options"]})
    manifest_path = args.output_dir / f"{stage}_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Stage source/configuration changed; choose a NEW AUDIT_RESULTS_DIR rather than mixing results")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log = args.output_dir / f"{stage}.log"
    for entry in entries:
        run_command(command(entry, args, path), log)
    rows = check_stage(path, entries, args)
    run_command([sys.executable, str(ROOT / "validate_results.py"), str(path)], log)
    run_command([sys.executable, str(ROOT / "summarize_results.py"), str(path),
                 "--output", str(args.output_dir / f"{stage}_summary.csv")], log)
    if stage == "baseline_pilot":
        groups = defaultdict(list)
        for row in rows:
            groups[row["config_id"]].append(row)
        winner = max(groups.values(), key=lambda group: sum(int(r["generated_tokens"]) for r in group)
                     / sum(float(r["total_time_s"]) for r in group))
        selected = dict(single=float(winner[0]["pipesd_single_threshold"]), multi=float(winner[0]["pipesd_multi_threshold"]),
                        code_sha256=code_hash(), pilot_sha256=file_hash(path),
                        rule="highest aggregate throughput on the 30-prompt pilot")
        (args.output_dir / "baseline_selected_thresholds.json").write_text(json.dumps(selected, indent=2) + "\n")
    write_report(args.output_dir)


def write_report(directory):
    lines = ["# Supervisor audits", "", "These results do not replace historical paper tables.",
             "Preview includes the first draft forward; it is not entirely additional overhead.",
             "Stored tensor-state bytes are not process RSS or full peak memory.", ""]
    fields = (("throughput_aggregate_tok_s", False), ("prompt_e2e_p95_s", False), ("slm_time_mean_s", False),
              ("verified_token_acceptance_rate", True), ("avg_gamma_mean", False), ("switch_rate_weighted", True),
              ("compression_on_ratio_mean", True), ("shadow_reuse_rate_weighted", True))
    def fmt(row, key, percent=False):
        value = row.get(key, "")
        return "--" if value == "" else f"{float(value) * (100 if percent else 1):.3f}" + ("%" if percent else "")
    for path in sorted(directory.glob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["workload"] == "ALL"]
        lines.extend([f"## {path.stem}", "",
                      "| Config | Method | N | token/s | P95 s | Draft s/prompt | Verified acc. | Avg gamma | Switch | Comp.-on (prompt avg) | Reuse |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            cells = [row["run_tag"], row["method"], row["prompts"]] + [fmt(row, key, pct) for key, pct in fields]
            lines.append("| " + " | ".join(cells) + " |")
        paired = defaultdict(dict)
        for row in rows:
            paired[row["run_tag"]][row["method"]] = row
        for tag, methods in paired.items():
            if {"dynamic_dssd", "async_sd"} <= set(methods):
                gain = (float(methods["async_sd"]["throughput_aggregate_tok_s"])
                        / float(methods["dynamic_dssd"]["throughput_aggregate_tok_s"]) - 1) * 100
                lines.append(f"- {tag}: RACE-SD aggregate gain over Dynamic DSSD = {gain:+.3f}% (point estimate).")
        if "timing" in path.stem:
            for row in rows:
                lines.append(f"- {row['method']}: preview {fmt(row, 'preview_time_per_fresh_round_ms')} ms/fresh round; "
                             f"controller {fmt(row, 'controller_decision_time_per_round_ms')} ms/round, "
                             f"{fmt(row, 'controller_time_fraction_e2e', True)} of E2E; cached previews={row['preview_cached_count']}.")
        if "precision" in path.stem or "quantization" in path.stem:
            for row in rows:
                lines.append(f"- {row['run_tag']}/{row['method']}: dtype={row['resolved_draft_dtype']}, "
                             f"quantization={row['draft_quantization']}, tensor-state bytes={row['draft_state_tensor_bytes']}, "
                             f"INT8 Linear layers={row['quantized_linear_count']}, engine={row['resolved_quantized_engine'] or '--'}.")
        lines.append("")
    (directory / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    for key in ("max_len", "cpu_threads", "cpu_interop_threads", "sensitivity_per_workload",
                "confirmation_per_workload", "precision_per_workload", "precision_repeats", "timing_per_workload"):
        if getattr(args, key) < 1:
            raise SystemExit(f"{key} must be positive")
    if args.stage == "plan":
        total = sum(print_plan(stage, plan(stage, args), args) for stage in REQUIRED)
        print(f"Required total: {total} generated runs plus unit tests; optional INT8/new-baseline stages excluded.")
        for stage in OPTIONAL[:-1]:
            print_plan(stage, plan(stage, args), args)
        print("baseline_main: 900 runs after pilot selection; about 2-4 h plus loading/warm-up.")
        print("Estimates exclude loading/warm-up. BF16/INT8 kernel speed must be measured on the server.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        write_report(args.output_dir)
        print(args.output_dir / "audit_report.md")
        return
    with (args.output_dir / ".audit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another audit is running in this output directory")
        run_command(["/bin/bash", str(ROOT / "capture_environment.sh"),
                     str(args.output_dir / f"environment_{args.stage}.txt")], args.output_dir / "environment.log")
        for stage in (REQUIRED if args.stage == "all" else (args.stage,)):
            run_stage(stage, args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"[FAIL] {exc}")
