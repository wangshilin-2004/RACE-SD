#!/usr/bin/env python3
"""Check audit plans, coverage/provenance, and optional CPU quantization."""

import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import supervisor_audits as audit


def arguments(test_case):
    # Use an ephemeral synthetic input, independent of private evaluation data.
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    prompt_path = Path(directory.name) / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"source": "unit", "text": "Synthetic test input."}) + "\n")
    with patch("sys.argv", ["audit", "plan", "--prompt-file", str(prompt_path)]):
        return audit.parse_args()


def batch_helpers(names, namespace=None):
    source = ast.parse((audit.ROOT / "batch_experiments.py").read_text())
    nodes = [node for node in source.body if getattr(node, "name", None) in names]
    ns = dict(argparse=argparse, hashlib=hashlib, json=json, copy=copy, Path=Path, platform=platform, math=math)
    ns.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "batch_experiments.py", "exec"), ns)
    return ns


def fixture(entries, args):
    rows = []
    for config, (tag, method, entry, overrides) in enumerate(audit.configurations(entries)):
        options = dict(audit.common_options(args), **entry["options"])
        options.update(overrides)
        for index in range(entry["per_workload"] * 3):
            row = dict(config_id=str(config), prompt_id=f"p{index}", input_hash=f"hash{index}",
                       status="ok", code_sha256=audit.code_hash(), run_tag=tag, method=method)
            for key, value in options.items():
                key = {"bandwidth": "bandwidth_mbps", "cpu_threads": "torch_num_threads",
                       "cpu_interop_threads": "torch_num_interop_threads", "method_order": "method_order_mode"}.get(key, key)
                if key in {"prompt_file", "controller_variant_file"}:
                    row[key + "_sha256"] = audit.file_hash(value)
                else:
                    row[key] = str(value)
            row["resolved_draft_dtype"] = "torch." + options["draft_model_dtype"]
            row["resolved_target_dtype"] = "torch.float16"
            rows.append(row)
    return rows


class AuditTest(unittest.TestCase):
    def test_required_run_counts(self):
        args = arguments(self)
        counts = {stage: sum(entry["per_workload"] * 3 for _, _, entry, _ in audit.configurations(audit.plan(stage, args)))
                  for stage in audit.REQUIRED}
        self.assertEqual(counts, dict(sanity=99, precision=240, timing=60, sensitivity=1710,
                                      sensitivity_poor=570, reference=420, action_space=180))

    def test_precision_order_is_balanced_and_paired(self):
        entries = audit.plan("precision", arguments(self))
        self.assertEqual([e["options"]["draft_model_dtype"] for e in entries],
                         ["float32", "bfloat16", "bfloat16", "float32"])
        self.assertTrue(all(set(e["methods"].split(",")) == {"dynamic_dssd", "async_sd"} for e in entries))
        self.assertEqual(len(set(e["tag"] for e in entries)), 4)

    def test_gamma_diagnosis_covers_precisions_lengths_and_repeats(self):
        entries = audit.plan("gamma_diagnosis", arguments(self))
        self.assertEqual(sum(e["per_workload"] * 3 * len(e["methods"].split(",")) for e in entries), 1020)
        self.assertEqual(len({e["tag"] for e in entries}), len(entries))
        for repeat in (1, 2):
            subset = [e for e in entries if e["tag"].startswith(f"gamma_r{repeat}_")]
            self.assertEqual(sum(e["methods"] == "ar_target" for e in subset), 1)
            for dtype in ("float32", "bfloat16"):
                fixed = [e for e in subset if e["methods"] == "optimized_dssd"
                         and e["options"]["draft_model_dtype"] == dtype]
                self.assertEqual([e["options"]["gamma"] for e in fixed],
                                 [1, 2, 3, 4, 6, 8] if repeat == 1 else [8, 6, 4, 3, 2, 1])
                adaptive = [e for e in subset if e["tag"].endswith(f"{dtype}_adaptive")]
                self.assertEqual(len(adaptive), 1)
                self.assertEqual(set(adaptive[0]["methods"].split(",")), {"dynamic_dssd", "async_sd"})

    def test_gamma_confirmation_covers_full_workload_and_preserves_adaptive_initial_gamma(self):
        args = arguments(self)
        entries = audit.plan("gamma_confirmation", args)
        self.assertEqual(sum(e["per_workload"] * 3 * len(e["methods"].split(",")) for e in entries), 3000)
        self.assertTrue(all(e["options"]["draft_model_dtype"] == "bfloat16" for e in entries))
        self.assertEqual([e["options"].get("gamma") for e in entries], [1, 2, None, None, 2, 1])
        for e in entries:
            if e["tag"].endswith("adaptive"):
                self.assertEqual(set(e["methods"].split(",")), {"ar_target", "dynamic_dssd", "async_sd"})
                self.assertNotIn("gamma", e["options"])

    def test_bf16_gamma_pilot_keeps_all_lengths_and_target_override(self):
        args = arguments(self)
        args.target_model = "/models/target-13b"
        entries = audit.plan("bf16_gamma_pilot", args)
        self.assertEqual(sum(e["per_workload"] * 3 * len(e["methods"].split(",")) for e in entries), 540)
        self.assertTrue(all(e["options"]["draft_model_dtype"] == "bfloat16" for e in entries))
        for repeat in (1, 2):
            fixed = [e["options"]["gamma"] for e in entries
                     if e["tag"].startswith(f"gamma_r{repeat}_") and e["methods"] == "optimized_dssd"]
            self.assertEqual(set(fixed), {1, 2, 3, 4, 6, 8})
        cmd = audit.command(entries[0], args, Path("/tmp/test-only.csv"))
        self.assertEqual(cmd[cmd.index("--target-model") + 1], args.target_model)

    def test_every_planned_cli_is_accepted(self):
        args = arguments(self)
        helpers = batch_helpers({"parse_args", "validate_args", "controller_variants"})
        for stage in audit.REQUIRED + audit.OPTIONAL[:-1]:
            for entry in audit.plan(stage, args):
                cmd = audit.command(entry, args, Path("/tmp/test-only.csv"))
                with patch("sys.argv", cmd[1:]):
                    cli = helpers["parse_args"]()
                helpers["validate_args"](cli)
                helpers["controller_variants"](cli)

    def test_reference_variants_change_one_scale_at_a_time(self):
        variants = json.loads((audit.ROOT / "controller_reference_variants.json").read_text())
        self.assertEqual(len(variants), 7)
        self.assertEqual(variants[0]["overrides"], {})
        self.assertTrue(all(len(v["overrides"]) == 1 for v in variants[1:]))

    def test_all_planned_configuration_groups_validate(self):
        args = arguments(self)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            for stage in audit.REQUIRED + audit.OPTIONAL[:-1]:
                entries = audit.plan(stage, args)
                rows = fixture(entries, args)
                for row in rows:
                    if row.get("draft_quantization") == "dynamic-int8":
                        row.update(quantized_linear_count="2", resolved_quantized_engine="qnnpack")
                fields = sorted(set().union(*(row.keys() for row in rows)))
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(len(audit.check_stage(path, entries, args)), len(rows))

    def test_shell_preserves_explicit_model_and_seed(self):
        import os
        import subprocess
        env = dict(os.environ, PYTHON="/bin/echo", DRAFT_MODEL="explicit-draft", SEED="42")
        output = subprocess.check_output(["/bin/bash", str(audit.ROOT / "run_supervisor_audits.sh"), "plan"],
                                         env=env, text=True)
        self.assertIn("--draft-model explicit-draft", output)
        self.assertIn("--seed 42", output)

    def test_coverage_and_stale_source_are_rejected(self):
        args = arguments(self)
        entries = audit.plan("timing", args)
        rows = fixture(entries, args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            def write(items):
                fields = sorted(set().union(*(row.keys() for row in rows)))
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(items)
            write(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(audit.check_stage(path, entries, args)), 60)
            write([row for row in rows if row["method"] != "async_sd"])
            with self.assertRaisesRegex(ValueError, "Coverage mismatch"):
                audit.check_stage(path, entries, args)
            rows[0]["code_sha256"] = "old"
            write(rows)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                audit.check_stage(path, entries, args)

    def test_wrong_dtype_and_prompt_pairing_are_rejected(self):
        args = arguments(self)
        entries = audit.plan("precision", args)
        rows = fixture(entries, args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            fields = sorted(set().union(*(row.keys() for row in rows)))
            def write():
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            rows[0]["resolved_draft_dtype"] = "torch.float16"
            write()
            with self.assertRaisesRegex(ValueError, "precision"):
                audit.check_stage(path, entries, args)
            rows[0]["resolved_draft_dtype"] = "torch.float32"
            rows[0]["input_hash"] = "different"
            write()
            with self.assertRaisesRegex(ValueError, "identical prompts"):
                audit.check_stage(path, entries, args)

    def test_quantization_rejects_wrong_device_and_dtype(self):
        helpers = batch_helpers({"parse_args", "validate_args"})
        for extra in (["--draft-quantization", "dynamic-int8", "--device-draft", "cuda:0"],
                      ["--draft-quantization", "dynamic-int8", "--draft-model-dtype", "bfloat16"]):
            with patch("sys.argv", ["batch", "--output", "unused", "--draft-model", "draft", "--target-model", "target"] + extra):
                cli = helpers["parse_args"]()
            with self.assertRaises(SystemExit):
                helpers["validate_args"](cli)

    def test_baseline_main_requires_real_pilot_selection(self):
        args = arguments(self)
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            with self.assertRaisesRegex(ValueError, "baseline_pilot"):
                audit.plan("baseline_main", args)

    def test_summary_retains_scales_and_true_slm_key(self):
        import summarize_results as summary
        row = dict(run_tag="synthetic", config_id="test", method="dynamic_dssd", draft_model="draft",
                   target_model="target", seed="321", gamma="4", dynamic_gamma_set="1,4",
                   lambda_sw="0.02", lambda_cmp="0.02", lambda_tail="0.04", bandwidth_mbps="100", rtt_ms="20",
                   beta="0.2", total_time_s="2", total_slm_s="0.25", generated_tokens="8", cmp_k_ref="128",
                   cmp_tau0="0.97", cmp_delta_tau="0.01", controller_decision_time_total_s="0.02")
        result = summary.aggregate([row], "ALL")
        self.assertEqual(result["slm_time_mean_s"], 0.25)
        self.assertEqual(result["cmp_k_ref"], "128")
        self.assertEqual(result["controller_time_fraction_e2e"], 0.01)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable; run on server")
class QuantizationTest(unittest.TestCase):
    def test_small_model_generation_with_bf16_and_int8(self):
        import torch
        from test_shadow_pipeline import core
        helpers = batch_helpers({"quantize_cpu_draft"}, {"torch": torch})
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(17, 16)
                self.head = torch.nn.Linear(16, 17)
            def forward(self, ids):
                return SimpleNamespace(logits=self.head(self.embedding(ids)))
        supported = set(torch.backends.quantized.supported_engines) - {"none"}
        modes = ["bfloat16"] + (["dynamic-int8"] if "qnnpack" in supported else [])
        old_engine = torch.backends.quantized.engine
        try:
            for mode in modes:
                draft = Model().eval()
                if mode == "bfloat16":
                    draft = draft.to(torch.bfloat16)
                else:
                    cli = SimpleNamespace(draft_quantization=mode, draft_quantized_engine="qnnpack")
                    with contextlib.redirect_stderr(io.StringIO()):
                        draft = helpers["quantize_cpu_draft"](draft, cli)
                for fn in (core.generate_dynamic_DSSD, core.generate_async_dynamic_DSSD):
                    a = core.parse_arguments([])
                    a.max_len, a.rtt, a.bandwidth = 7, 0.0, 10000
                    a.dynamic_warmup_rounds = 1
                    a.dynamic_gamma_set = "1,2"
                    a.return_runtime_stats = True
                    uav = core.UAVNode(draft, torch.device("cpu"), a)
                    bs = core.BSNode(Model().eval(), torch.device("cpu"), a)
                    tokenizer = SimpleNamespace(vocab_size=17, decode=lambda ids, **kw: str(ids.tolist()))
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = fn(uav, bs, torch.tensor([[1, 2, 3]]), tokenizer, a)
                    self.assertAlmostEqual(result[1] * result[2], 7)
        finally:
            torch.backends.quantized.engine = old_engine

    def test_real_linear_quantization_and_packed_memory(self):
        import torch
        helpers = batch_helpers({"quantize_cpu_draft", "state_tensor_bytes"}, {"torch": torch})
        supported = set(torch.backends.quantized.supported_engines) - {"none"}
        preferred = ("qnnpack",) if platform.machine().lower() in {"aarch64", "arm64"} else ("x86", "fbgemm", "qnnpack")
        engine = next((e for e in preferred if e in supported), None)
        if engine is None:
            self.skipTest("No applicable CPU INT8 backend")
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)).eval()
        cli = SimpleNamespace(draft_quantization="dynamic-int8", draft_quantized_engine=engine)
        old_engine = torch.backends.quantized.engine
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                quantized = helpers["quantize_cpu_draft"](model, cli)
            output = quantized(torch.randn(2, 16))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(cli.quantized_linear_count, 2)
            self.assertGreater(helpers["state_tensor_bytes"](quantized), 0)
            self.assertLess(helpers["state_tensor_bytes"](quantized), helpers["state_tensor_bytes"](model))
        finally:
            torch.backends.quantized.engine = old_engine


if __name__ == "__main__":
    unittest.main()
