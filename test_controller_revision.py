#!/usr/bin/env python3
"""Dependency-free controller tests plus optional torch pipeline checks."""

import argparse
import ast
import copy
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Dict, List
import unittest
from unittest.mock import patch
import tempfile


ROOT = Path(__file__).resolve().parent


def load_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing definitions: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


ns = load_definitions("dynamic_dssd_main.py", {"DynamicDSSDController"}, {
    "argparse": argparse, "math": math, "Dict": Dict,
})
Controller = ns["DynamicDSSDController"]

link_ns = load_definitions(
    "dynamic_dssd_main.py",
    {"TimeVaryingLinkReplay", "ControllerLinkTelemetry"},
    {"argparse": argparse, "json": json, "time": time, "Dict": Dict, "List": List},
)
TimeVaryingLinkReplay = link_ns["TimeVaryingLinkReplay"]
ControllerLinkTelemetry = link_ns["ControllerLinkTelemetry"]


def controller(**kwargs):
    return Controller(SimpleNamespace(**kwargs))


def state():
    return {"difficulty": 0.3, "jitter": 0.0, "vocab_size": 32000,
            "tx_delay_fn": lambda size: 0.01 + size / 12500000}


def observation(**kwargs):
    result = dict(gamma=4, correct_num=2, flag=0, j=3, difficulty=0.3,
                  t_slm=0.02, t_llm=0.08, compression_attempted=True,
                  reject_topk=64, reject_mass_threshold=0.995,
                  compressed_feedback=True, candidate_kept_mass=0.999,
                  candidate_tail_mass=0.001, saving=0.9,
                  generated_tokens=3, elapsed_s=0.15)
    result.update(kwargs)
    return result


class ControllerRevisionTest(unittest.TestCase):
    def test_causal_link_telemetry_updates_after_completed_round(self):
        replay = TimeVaryingLinkReplay(
            [
                {"time_s": 0, "bw_up_mbps": 100, "bw_down_mbps": 200, "rtt_ms": 20},
                {"time_s": 1, "bw_up_mbps": 10, "bw_down_mbps": 20, "rtt_ms": 100},
            ],
            default_rtt_s=0.02,
            default_bw_mbps=100,
        )
        telemetry = ControllerLinkTelemetry(
            replay, SimpleNamespace(controller_link_mode="causal_previous_round")
        )
        initial_delay = telemetry.delay_bytes(1_000_000, "up")
        replay.start_time -= 2.0

        replay.delay_bytes(1_000_000, "up")
        replay.delay_bytes(1_000_000, "down")
        self.assertAlmostEqual(telemetry.delay_bytes(1_000_000, "up"), initial_delay)

        telemetry.observe_completed_round()
        self.assertAlmostEqual(telemetry.delay_bytes(1_000_000, "up"), 0.85)
        self.assertEqual(telemetry.stats()["controller_link_observation_count"], 1)

    def test_instantaneous_controller_queries_do_not_create_transfer_observations(self):
        replay = TimeVaryingLinkReplay(
            [{"time_s": 0, "bw_mbps": 100, "rtt_ms": 20}],
            default_rtt_s=0.02,
            default_bw_mbps=100,
        )
        telemetry = ControllerLinkTelemetry(
            replay, SimpleNamespace(controller_link_mode="instantaneous")
        )
        telemetry.delay_bytes(1024, "down")
        self.assertEqual(replay.last_transfer_states, {})

    def test_default_modes_remain_legacy(self):
        c = controller()
        self.assertEqual((c.score_mode, c.compression_estimator, c.acceptance_predictor),
                         ("ratio", "legacy", "legacy"))

    def test_legacy_score_matches_original_formula(self):
        c = controller()
        c.update(observation())
        a = c.prev_action
        alpha, adv, pcmp, cost = c.predicted_terms(a, state())
        risk = c.lambda_cmp * (1 - c.kept_mass_ema + c.tail_mass_ema + 1 - pcmp)
        risk += c.lambda_tail * (0.7 * (1-alpha**2) + 0.3 * (1-alpha**a["gamma"]))
        self.assertAlmostEqual(c.score(a, state()), adv / cost - risk)

    def test_difference_uses_cumulative_ratio_not_mean_of_ratios(self):
        c = controller(dynamic_score_mode="difference")
        c.update(observation(generated_tokens=2, elapsed_s=1))
        c.update(observation(generated_tokens=5, elapsed_s=4))
        self.assertEqual(c.throughput_estimate, 1.25)

    def test_invalid_time_is_rejected(self):
        c = controller(dynamic_score_mode="difference")
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.update(observation(elapsed_s=value))

    def test_difference_scale_and_penalty_units(self):
        c = controller(dynamic_score_mode="difference", dynamic_lambda_sw=0,
                       dynamic_lambda_cmp=0, dynamic_lambda_tail=0)
        c.throughput_estimate = 9.0
        c.score_reference_time = 0.25
        _, adv, _, cost = c.predicted_terms(c.prev_action, state())
        self.assertAlmostEqual(c.score(c.prev_action, state()), (adv-9*cost)/0.25)

    def test_hysteresis_remains_positive_at_zero_score(self):
        c = controller(dynamic_score_mode="difference")
        c.throughput_estimate = 10
        c.score_reference_time = 0.2
        self.assertAlmostEqual(c.hysteresis_margin(0), 0.1)
        self.assertEqual(c.hysteresis_margin(-1e-10), c.hysteresis_margin(1e-10))

    def test_history_cold_start_does_not_depend_on_action_boundaries(self):
        c = controller(dynamic_compression_estimator="history")
        for k, tau in ((16, 0.98), (512, 0.9999), (7, 0.8)):
            self.assertEqual(c.predict_compression_success(0.3, k, tau), 0.5)

    def test_history_updates_only_executed_configuration(self):
        c = controller(dynamic_compression_estimator="history")
        c.update(observation())
        self.assertAlmostEqual(c.predict_compression_success(0.3, 64, 0.995), 0.6)
        self.assertEqual(c.predict_compression_success(0.3, 32, 0.995), 0.5)
        self.assertEqual(c.predict_compression_success(0.3, 64, 0.98), 0.5)
        c.update(observation(compressed_feedback=False))
        self.assertAlmostEqual(c.predict_compression_success(0.3, 64, 0.995), 0.48)
        self.assertAlmostEqual(c.compression_brier_sum, 0.25 + 0.36)

    def test_history_does_not_count_accept_all_or_disabled_compression(self):
        c = controller(dynamic_compression_estimator="history")
        c.update(observation(flag=1, correct_num=4, j=5))
        c.update(observation(compression_attempted=False))
        self.assertEqual(c.compression_observation_count, 0)
        self.assertEqual(c.compression_history, {})

    def test_probability_stays_bounded(self):
        c = controller(dynamic_compression_estimator="history")
        for i in range(100):
            c.update(observation(compressed_feedback=i % 3 == 0))
            self.assertTrue(0 <= c.predict_compression_success(0.9, 64, 0.995) <= 1)

    def test_lcb_predictor_removes_blending(self):
        c = controller(dynamic_acceptance_predictor="lcb")
        c.update(observation())
        for difficulty in (0.0, 0.5, 1.0):
            self.assertEqual(c.predict_alpha(difficulty, 0), c.alpha_lcb())

    def test_difference_cold_start_uses_safe_warmup(self):
        c = controller(dynamic_score_mode="difference", dynamic_warmup_rounds=0)
        self.assertEqual(c.choose_action(state())["gamma"], c.gamma_safe)
        self.assertEqual(c.warmup_count, 1)

    def test_all_modes_produce_finite_scores(self):
        for score in ("ratio", "difference"):
            for estimate in ("legacy", "history"):
                c = controller(dynamic_score_mode=score, dynamic_compression_estimator=estimate,
                               dynamic_warmup_rounds=1)
                c.choose_action(state())
                for i in range(10):
                    c.update(observation(generated_tokens=3*(i+1), elapsed_s=0.15*(i+1)))
                    a = c.choose_action(state())
                    self.assertIn(a["gamma"], c.gamma_set)
                    self.assertTrue(math.isfinite(c.score(a, state())))

    def test_matrix_is_isolated_and_order_rotates(self):
        helpers = load_definitions("batch_experiments.py", {"controller_variants", "ordered_runs"}, {"copy": copy})
        cli = SimpleNamespace(controller_matrix=True, controller_score="ratio",
                              compression_estimator="legacy", acceptance_predictor="legacy", run_tag="pilot")
        variants = helpers["controller_variants"](cli)
        self.assertEqual(len(variants), 5)
        self.assertEqual(cli.run_tag, "pilot")
        first_positions = []
        for i in range(5):
            runs = helpers["ordered_runs"](variants, ["async_sd"], i, "rotate")
            first_positions.append(runs[0][0])
        self.assertEqual(len(set(first_positions)), 5)

    def test_oat_variant_file_has_unique_complete_configs(self):
        helpers = load_definitions(
            "batch_experiments.py", {"controller_variants"},
            {"copy": copy, "Path": Path, "json": json, "validate_args": lambda _: None},
        )
        cli = SimpleNamespace(
            controller_variant_file=str(ROOT / "controller_oat_variants.json"),
            controller_matrix=False,
            run_tag="oat",
            z_alpha=0.7,
            alpha_weight_history=0.65,
            alpha_weight_difficulty=0.20,
            alpha_weight_early_reject=0.15,
            cmp_weight_mass=0.45,
            cmp_weight_saving=0.25,
            cmp_weight_topk=0.20,
            cmp_weight_easiness=0.10,
            cmp_threshold_penalty=0.15,
            cmp_k_ref=64.0,
            cmp_tau0=0.98,
            cmp_delta_tau=0.02,
            lambda_sw=0.02,
            lambda_cmp=0.02,
            lambda_tail=0.04,
        )
        variants = helpers["controller_variants"](cli)
        names = [name for name, _ in variants]
        self.assertEqual(len(variants), 19)
        self.assertEqual(len(set(names)), 19)
        self.assertEqual(names[0], "baseline")
        self.assertEqual(variants[0][1].z_alpha, 0.7)

    def test_parser_exposes_modes(self):
        parser_ns = load_definitions("dynamic_dssd_main.py", {"parse_arguments"}, {"argparse": argparse})
        a = parser_ns["parse_arguments"](["--dynamic_score_mode", "difference",
                                          "--dynamic_compression_estimator", "history",
                                          "--controller_link_mode", "causal_previous_round"])
        self.assertEqual(a.dynamic_score_mode, "difference")
        self.assertEqual(a.dynamic_compression_estimator, "history")
        self.assertEqual(a.controller_link_mode, "causal_previous_round")

    def test_config_hash_distinguishes_algorithm_options(self):
        helpers = load_definitions("batch_experiments.py",
                                   {"parse_args", "config_id", "controller_variants"},
                                   {"argparse": argparse, "hashlib": hashlib, "json": json, "copy": copy})
        with patch("sys.argv", ["batch", "--output", "/tmp/not-written.csv",
                                "--draft-model", "draft", "--target-model", "target",
                                "--controller-matrix"]):
            cli = helpers["parse_args"]()
        hashes = []
        for _, variant in helpers["controller_variants"](cli):
            variant.run_tag = "same-tag"
            hashes.append(helpers["config_id"](variant, "async_sd"))
        self.assertEqual(len(set(hashes)), 5)
        instantaneous = copy.deepcopy(cli)
        instantaneous.controller_link_mode = "instantaneous"
        self.assertNotEqual(
            helpers["config_id"](cli, "async_sd"),
            helpers["config_id"](instantaneous, "async_sd"),
        )

    def test_result_checker_rejects_missing_variants_and_stale_code(self):
        import check_controller_revision as checker
        rows = []
        for i, variant in enumerate(sorted(checker.VARIANTS)):
            rows.append(dict(config_id=str(i), prompt_id="synthetic", input_hash="test-only",
                             method="async_sd", status="ok", error="", code_sha256=checker.current_code_hash(),
                             controller_score=variant[0], compression_estimator=variant[1],
                             acceptance_predictor=variant[2], controller_throughput_estimate_final="1",
                             controller_compression_observations="1", controller_compression_history_pairs="1",
                             controller_compression_brier_mean="0.25", reject_rounds="1",
                             total_time_s="1", generated_tokens="1"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            def write_fixture(items):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(items)
            write_fixture(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(checker.check(path, 1, {"async_sd"})), 5)
            write_fixture(rows[:-1])
            with self.assertRaisesRegex(ValueError, "Expected 5 rows"):
                checker.check(path, 1, {"async_sd"})
            rows[0]["code_sha256"] = "stale"
            write_fixture(rows)
            with self.assertRaisesRegex(ValueError, "code hash"):
                checker.check(path, 1, {"async_sd"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable; run on server")
class PipelineRevisionTest(unittest.TestCase):
    def test_tiny_model_both_generation_paths(self):
        import contextlib
        import io
        from test_shadow_pipeline import TinyModel, core, torch
        tokenizer = SimpleNamespace(vocab_size=17, decode=lambda ids, **kw: str(ids.tolist()))
        for fn in (core.generate_dynamic_DSSD, core.generate_async_dynamic_DSSD):
            for score, compression, acceptance in (
                ("ratio", "legacy", "legacy"), ("difference", "legacy", "legacy"),
                ("ratio", "history", "legacy"), ("difference", "history", "legacy"),
                ("difference", "history", "lcb"),
            ):
                with self.subTest(fn=fn.__name__, score=score, compression=compression, acceptance=acceptance):
                    a = core.parse_arguments([])
                    a.dynamic_score_mode, a.dynamic_compression_estimator = score, compression
                    a.dynamic_acceptance_predictor = acceptance
                    a.max_len, a.rtt, a.bandwidth = 7, 0.0, 10000
                    a.dynamic_warmup_rounds = 1
                    a.dynamic_gamma_set = "1,2"
                    a.return_runtime_stats = True
                    uav = core.UAVNode(TinyModel(), torch.device("cpu"), a)
                    bs = core.BSNode(TinyModel(), torch.device("cpu"), a)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = fn(uav, bs, torch.tensor([[1, 2, 3]]), tokenizer, a)
                    self.assertAlmostEqual(result[1] * result[2], 7)
                    if score == "difference":
                        self.assertGreater(result[9]["controller_throughput_estimate_final"], 0)


if __name__ == "__main__":
    unittest.main()
