#!/usr/bin/env python3
"""CPU-only audit tests, including a gate-pass/zero-residual counterexample."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = object
    stub.AutoModelForCausalLM = object
    sys.modules["transformers"] = stub

import dynamic_dssd_main as core
from residual_sanity import ResidualAudit, instrument, summarize


def fake_batch(packet, q):
    def run(method):
        node = object.__new__(core.UAVNode)
        token = node.resample_DSSD(1, packet, q.unsqueeze(0))
        stats = dict(reject_compressed_cnt=int(packet["mode"] == "compressed"),
                     reject_full_cnt=int(packet["mode"] == "full"))
        return {"token": token}, stats
    return SimpleNamespace(core=core, run_method=run, make_row=lambda: {})


class CounterTests(unittest.TestCase):
    def test_threshold_and_packet_denominators(self):
        audit = ResidualAudit()
        audit.observe("compressed", 0, 1)
        audit.observe("compressed", 1e-12, 1)
        audit.observe("compressed", 1.01e-12, 1)
        audit.observe("full", 0, 0)
        fields = audit.fields()
        self.assertEqual(fields["residual_compressed_fallback_rate"], 2 / 3)
        self.assertEqual(fields["residual_compressed_exact_zero"], 1)
        self.assertEqual(fields["residual_compressed_small_positive"], 1)
        self.assertEqual(fields["residual_full_argmax"], 1)
        audit.reset()
        self.assertEqual(audit.fields()["residual_full_fallback_rate"], "")

    def test_nonfinite_is_error_not_silent_zero(self):
        for mass in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                ResidualAudit().observe("full", mass, 1)

    def test_summary_uses_pooled_counts_and_separates_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            rows = []
            for index, (calls, hits) in enumerate(((1, 1), (9, 0))):
                audit = ResidualAudit()
                for step in range(calls):
                    audit.observe("compressed", 0 if step < hits else 0.1, 1)
                rows.append(dict(audit.fields(), diagnostic_only=1, config_id="test",
                                 prompt_id=str(index), method="optimized_dssd", status="ok",
                                 target_model="toy", workload=f"workload_{index}"))
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                out = summarize(path)
            with out.open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            pooled = next(row for row in summary if row["workload"] == "ALL"
                          and row["packet_mode"] == "compressed")
            self.assertEqual(float(pooled["fallback_rate"]), 0.1)
            self.assertEqual(int(pooled["affected_prompt_runs"]), 1)
            self.assertEqual(len(summary), 6)


class SamplingTests(unittest.TestCase):
    def test_high_retained_mass_does_not_ensure_positive_residual(self):
        p = torch.tensor([0.996, 0.004])
        q = torch.tensor([0.997, 0.003])
        args = SimpleNamespace(compress_reject_packet=True, reject_packet_dtype="float16",
                               reject_topk=1, reject_mass_threshold=0.995, reject_min_prob=0)
        packet = core.build_reject_packet(p, args)
        self.assertEqual(packet["mode"], "compressed")
        self.assertGreater(float((p - q).clamp(min=0).sum()), 0)
        batch = fake_batch(packet, q)
        with instrument(batch) as audit:
            metrics, _ = batch.run_method("optimized_dssd")
            self.assertEqual(audit.counts["compressed"]["fallback"], 1)
            self.assertEqual(int(metrics["token"].item()), 0)

    def test_instrumentation_preserves_tokens_rng_and_inputs(self):
        q = torch.tensor([0.4, 0.3, 0.3])
        packets = [
            dict(mode="full", probs=torch.tensor([0.6, 0.1, 0.3])),
            dict(mode="full", probs=q.clone()),
            dict(mode="full", probs=torch.zeros(3)),
            dict(mode="compressed", ids=torch.tensor([0, 1]), vocab_size=3,
                 probs=torch.tensor([0.6, 0.4])),
            dict(mode="compressed", ids=torch.tensor([0]), vocab_size=3,
                 probs=torch.tensor([0.3])),
        ]
        original_resample = core.UAVNode.resample_DSSD
        for packet in packets:
            with self.subTest(mode=packet["mode"], probs=packet["probs"]):
                batch = fake_batch(packet, q)
                original_run = batch.run_method
                p_before, q_before = packet["probs"].clone(), q.clone()
                torch.manual_seed(321)
                expected = [batch.run_method("optimized_dssd")[0]["token"] for _ in range(10)]
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(321)
                with instrument(batch) as audit:
                    actual = [batch.run_method("optimized_dssd")[0]["token"] for _ in range(10)]
                    self.assertEqual(audit.counts[packet["mode"]]["calls"], 1)
                    self.assertEqual(batch.make_row()["diagnostic_only"], 1)
                self.assertTrue(torch.equal(torch.stack(expected), torch.stack(actual)))
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(p_before, packet["probs"]))
                self.assertTrue(torch.equal(q_before, q))
                self.assertIs(core.UAVNode.resample_DSSD, original_resample)
                self.assertIs(batch.run_method, original_run)

    def test_restores_wrappers_after_error(self):
        batch = fake_batch(dict(mode="full", probs=torch.tensor([1.0])), torch.tensor([1.0]))
        original = core.UAVNode.resample_DSSD
        with self.assertRaises(RuntimeError):
            with instrument(batch):
                raise RuntimeError("test")
        self.assertIs(core.UAVNode.resample_DSSD, original)

    def test_count_mismatch_is_detected(self):
        batch = fake_batch(dict(mode="full", probs=torch.tensor([1.0])), torch.tensor([1.0]))
        batch.run_method = lambda method: ({}, {"reject_full_cnt": 1})
        with instrument(batch):
            with self.assertRaises(RuntimeError):
                batch.run_method("vanilla_dssd")

    def test_ideal_distortion_equality(self):
        p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
        q = torch.tensor([0.4, 0.25, 0.35], dtype=torch.float64)
        residual = (p - q).clamp(min=0)
        compressed = residual.clone()
        compressed[1:] = 0
        tv = 0.5 * (residual / residual.sum() - compressed / compressed.sum()).abs().sum()
        self.assertAlmostEqual(float(tv), float(residual[1:].sum() / residual.sum()))


class RunnerTests(unittest.TestCase):
    def test_actual_batch_entry_with_tiny_cpu_models(self):
        import batch_experiments as batch
        import residual_sanity

        class TinyModel(torch.nn.Module):
            def __init__(self, phase):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                self.phase = phase
                self.config = SimpleNamespace(vocab_size=17)

            def forward(self, input_ids):
                base = torch.arange(17, device=input_ids.device, dtype=torch.float32)
                offsets = input_ids.float().cumsum(dim=1).unsqueeze(-1) * 0.17
                return SimpleNamespace(logits=3 * torch.cos(base + offsets + self.phase))

        tokenizer = SimpleNamespace(
            get_vocab=lambda: {str(i): i for i in range(17)},
            encode=lambda text, return_tensors: torch.tensor([[1, 2, 3]]),
            decode=lambda ids, skip_special_tokens: "toy output",
        )
        model_loader = SimpleNamespace(from_pretrained=lambda path, **kw:
                                       TinyModel(0 if path == "draft" else 2))
        tokenizer_loader = SimpleNamespace(from_pretrained=lambda path: tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            prompts.write_text("".join(json.dumps(dict(source=source, text="toy")) + "\n"
                                       for source in ("a", "b", "c")))
            output = Path(directory) / "diagnostic.csv"
            argv = ["residual_sanity.py", "--prompt-file", str(prompts), "--output", str(output),
                    "--draft-model", "draft", "--target-model", "target",
                    "--device-draft", "cpu", "--device-target", "cpu", "--max-len", "8",
                    "--rtt-ms", "0", "--bandwidth", "1000000",
                    "--reject-topk", "1", "--mass-threshold", "0",
                    "--methods", "vanilla_dssd,optimized_dssd,async_sd"]
            with patch.object(batch, "AutoModelForCausalLM", model_loader), \
                    patch.object(batch, "AutoTokenizer", tokenizer_loader), \
                    patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                residual_sanity.main()
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertTrue(all(row["diagnostic_only"] == "1" for row in rows))
            self.assertTrue(output.with_name("diagnostic_residual_summary.csv").is_file())
            self.assertGreater(sum(int(row["residual_compressed_calls"]) for row in rows), 0)
            self.assertGreater(sum(int(row["residual_full_calls"]) for row in rows), 0)


if __name__ == "__main__":
    unittest.main()
