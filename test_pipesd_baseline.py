#!/usr/bin/env python3
"""CPU-only checks for the PipeSD control and scheduling port."""

import contextlib
import importlib.util
import io
from types import SimpleNamespace
import unittest

from pipesd_utils import (
    pipesd_dynamic_token_schedule,
    pipesd_should_verify,
)


class PipeSDTriggerTest(unittest.TestCase):
    def test_single_threshold_triggers(self):
        self.assertTrue(pipesd_should_verify([0.99, 0.93], 0.94, 0.80))

    def test_product_threshold_triggers(self):
        self.assertTrue(pipesd_should_verify([0.96, 0.95, 0.95], 0.90, 0.90))

    def test_confident_prefix_continues(self):
        self.assertFalse(pipesd_should_verify([0.98, 0.98], 0.94, 0.90))


class PipeSDSchedulerTest(unittest.TestCase):
    def test_empty_schedule(self):
        self.assertEqual(pipesd_dynamic_token_schedule([], 0.02, 0.01), ([], 0.0))

    def test_schedule_covers_tokens_once(self):
        batches, completion = pipesd_dynamic_token_schedule(
            [0.03, 0.03, 0.03], 0.02, 0.01
        )
        self.assertEqual([index for batch in batches for index in batch], [0, 1, 2])
        self.assertGreaterEqual(completion, 0.10)

    def test_batching_reduces_fixed_overhead(self):
        batches, _ = pipesd_dynamic_token_schedule(
            [0.001, 0.001, 0.001, 0.001], 0.05, 0.001
        )
        self.assertLess(len(batches), 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable; run on server")
class PipeSDGenerationTest(unittest.TestCase):
    def test_generation_and_round_accounting(self):
        import torch
        from test_shadow_pipeline import TinyModel, core
        from external_baselines import generate_pipesd

        args = core.parse_arguments([])
        args.max_len, args.rtt, args.bandwidth = 13, 2.0, 10000
        args.pipesd_min_gamma, args.pipesd_max_gamma = 1, 4
        args.pipesd_single_threshold, args.pipesd_multi_threshold = 0.94, 0.90
        args.pipesd_packet_dtype = "float16"
        uav = core.UAVNode(TinyModel(), torch.device("cpu"), args)
        bs = core.BSNode(TinyModel(), torch.device("cpu"), args)
        tokenizer = SimpleNamespace(vocab_size=17, decode=lambda ids, **kw: str(ids.tolist()))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = generate_pipesd(uav, bs, torch.tensor([[1, 2, 3]]), tokenizer, args)
        stats, rounds = result[-1], result[6]
        self.assertAlmostEqual(result[1] * result[2], args.max_len)
        self.assertGreater(rounds, 0)
        self.assertTrue(0 <= result[3] <= 1)
        self.assertEqual(stats["accept_all_rounds"] + stats["reject_rounds"], rounds)
        self.assertEqual(stats["pipesd_waiting_launches"], stats["pipesd_waiting_hits"]
                         + stats["pipesd_waiting_discard_reject"] + stats["pipesd_waiting_discard_mismatch"])
        for component in ("total", "comm", "slm", "llm"):
            self.assertEqual(len(stats[f"round_{component}_latency_samples_ms"]), rounds)


if __name__ == "__main__":
    unittest.main()
