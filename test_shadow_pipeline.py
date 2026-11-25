#!/usr/bin/env python3
"""Small CPU-only checks for transactional shadow drafting."""

import threading
import sys
import types
import unittest
from types import SimpleNamespace

import torch

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object
    transformers_stub.AutoModelForCausalLM = object
    sys.modules["transformers"] = transformers_stub

import dynamic_dssd_main as core


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=17):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, length = input_ids.shape
        base = torch.arange(
            self.vocab_size, device=input_ids.device, dtype=torch.float32
        )
        offsets = input_ids.to(torch.float32).cumsum(dim=1).unsqueeze(-1)
        logits = torch.cos(base.view(1, 1, -1) + offsets * 0.17)
        return SimpleNamespace(logits=logits.reshape(batch, length, self.vocab_size))


def args(seed=123):
    return SimpleNamespace(
        seed=seed,
        temperature=0.7,
        top_k=0,
        top_p=0.0,
        use_dist_summary=False,
        async_shadow_anchor_mode="sample",
    )


def controller_args(**overrides):
    values = {
        "dynamic_min_gamma": 1,
        "dynamic_gamma_set": "1,2,3,4,6,8",
        "dynamic_reject_topk_set": "16,32,64,128",
        "dynamic_reject_mass_set": "0.98,0.99,0.995",
        "reject_packet_dtype": "float16",
        "dynamic_joint_packet_control": True,
        "dynamic_beta": 0.2,
        "dynamic_z_alpha": 0.7,
        "dynamic_lambda_sw": 0.02,
        "dynamic_lambda_cmp": 0.02,
        "dynamic_lambda_tail": 0.04,
        "dynamic_hysteresis": 0.01,
        "dynamic_min_residency": 2,
        "dynamic_alpha_panic": 0.18,
        "dynamic_safe_gamma": 1,
        "dynamic_downlink_budget_frac": 0.45,
        "dynamic_warmup_rounds": 2,
        "dynamic_panic_reject_rate": 0.8,
        "dynamic_panic_early_reject": 0.7,
        "dynamic_debug": False,
        "compress_reject_packet": True,
        "reject_topk": 64,
        "reject_mass_threshold": 0.995,
        "gamma": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ShadowPipelineTest(unittest.TestCase):
    def test_map_anchor_uses_top1_without_advancing_shadow_rng(self):
        prefix = torch.tensor([[1, 4, 2]], dtype=torch.long)
        map_args = args()
        map_args.async_shadow_anchor_mode = "map"
        node = core.UAVNode(TinyModel(), torch.device("cpu"), map_args)
        rng_before = node.shadow_rng.get_state().clone()

        info = node.sample_next_token_info(prefix)
        with torch.no_grad():
            expected = torch.argmax(node.model(prefix).logits[:, -1, :], dim=-1)

        self.assertEqual(info["mode"], "map")
        self.assertEqual(info["token_id"], int(expected.item()))
        self.assertTrue(torch.equal(rng_before, node.shadow_rng.get_state()))

    def test_preview_logits_match_normal_first_draft_step(self):
        prefix = torch.tensor([[1, 4, 2]], dtype=torch.long)
        reused_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())
        normal_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())

        preview = reused_node.preview_difficulty(prefix)
        reused = reused_node.draft_DSSD(
            prefix, gamma=3, first_logits=preview["first_logits"]
        )
        normal = normal_node.draft_DSSD(prefix, gamma=3)

        self.assertTrue(torch.equal(reused[0], normal[0]))
        self.assertEqual(reused[1], normal[1])
        self.assertTrue(torch.equal(reused[2], normal[2]))
        self.assertTrue(
            torch.equal(reused_node.main_rng.get_state(), normal_node.main_rng.get_state())
        )

    def test_reuse_matches_normal_draft_and_commits_rng(self):
        prefix = torch.tensor([[1, 4, 2]], dtype=torch.long)
        shadow_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())
        normal_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())

        shadow = shadow_node.prepare_shadow_candidate(prefix, shadow_gamma=4)
        expected = normal_node.draft_DSSD(shadow["optimistic_prefix"], gamma=2)
        actual = shadow_node.materialize_shadow_for_gamma(shadow, gamma=2)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(
            torch.equal(shadow_node.main_rng.get_state(), normal_node.main_rng.get_state())
        )

    def test_short_shadow_extends_to_larger_requested_gamma(self):
        prefix = torch.tensor([[1, 4, 2]], dtype=torch.long)
        shadow_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())
        normal_node = core.UAVNode(TinyModel(), torch.device("cpu"), args())

        shadow = shadow_node.prepare_shadow_candidate(prefix, shadow_gamma=1)
        expected = normal_node.draft_DSSD(shadow["optimistic_prefix"], gamma=3)
        actual = shadow_node.materialize_shadow_for_gamma(shadow, gamma=3)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(
            torch.equal(shadow_node.main_rng.get_state(), normal_node.main_rng.get_state())
        )

    def test_cached_distribution_reproduces_preview_difficulty(self):
        prefix = torch.tensor([[1, 4, 2]], dtype=torch.long)
        node = core.UAVNode(TinyModel(), torch.device("cpu"), args())
        shadow = node.prepare_shadow_candidate(prefix, shadow_gamma=2)

        cached = node.difficulty_from_probs(shadow["q_probs"][0])
        direct = node.preview_difficulty(shadow["optimistic_prefix"])
        for key in ("entropy_norm", "qmax", "margin", "difficulty"):
            self.assertAlmostEqual(cached[key], direct[key], places=6)

    def test_pre_cancelled_shadow_does_not_start(self):
        event = threading.Event()
        event.set()
        node = core.UAVNode(TinyModel(), torch.device("cpu"), args())
        result = node.prepare_shadow_candidate(
            torch.tensor([[1, 2]], dtype=torch.long), 3, event
        )
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["shadow_compute_time_s"], 0.0)


class ControllerTest(unittest.TestCase):
    def test_core_defaults_include_low_gamma(self):
        parsed = core.parse_arguments([])
        self.assertEqual(parsed.dynamic_gamma_set, "1,2,3,4,6,8")
        self.assertEqual(parsed.dynamic_min_gamma, 1)
        self.assertEqual(parsed.dynamic_safe_gamma, 1)
        self.assertEqual(parsed.async_shadow_anchor_mode, "sample")
        self.assertEqual(parsed.async_shadow_overlap_margin, 1.0)

    def test_warmup_uses_safe_gamma_before_scoring(self):
        controller = core.DynamicDSSDController(controller_args())
        action = controller.choose_action({})
        self.assertEqual(action["gamma"], 1)
        self.assertEqual(controller.warmup_count, 1)

    def test_acceptance_ema_ignores_unverified_suffix(self):
        controller = core.DynamicDSSDController(controller_args())
        controller.update({
            "gamma": 8,
            "correct_num": 1,
            "flag": 0,
            "j": 2,
            "difficulty": 0.5,
            "t_slm": 0.08,
            "t_llm": 0.10,
            "kept_mass": 1.0,
            "tail_mass": 0.0,
            "saving": 0.0,
            "compression_attempted": True,
        })
        # One accepted token followed by one rejection is 1/2 over verified
        # proposals, not 1/8 over the originally drafted block.
        self.assertAlmostEqual(controller.last_round_alpha, 0.5)
        self.assertAlmostEqual(controller.alpha_ema, 0.58)

    def test_safe_mode_preserves_gated_compression_and_counts_gamma_switch(self):
        controller = core.DynamicDSSDController(controller_args(
            dynamic_warmup_rounds=0,
            dynamic_alpha_panic=0.5,
            dynamic_panic_reject_rate=0.5,
            dynamic_panic_early_reject=0.5,
        ))
        controller.alpha_ema = 0.1
        controller.reject_rate_ema = 1.0
        controller.early_reject_ema = 1.0
        action = controller.choose_action({"difficulty": 0.5})
        self.assertEqual(action["gamma"], 1)
        self.assertTrue(action["compress_reject_packet"])
        self.assertEqual(controller.switch_count, 1)

    def test_compression_ema_uses_candidate_mass_only_on_reject(self):
        controller = core.DynamicDSSDController(controller_args())
        common = {
            "gamma": 1,
            "correct_num": 0,
            "j": 1,
            "difficulty": 0.5,
            "t_slm": 0.01,
            "t_llm": 0.02,
            "kept_mass": 1.0,
            "tail_mass": 0.0,
            "saving": 0.0,
        }
        controller.update({
            **common,
            "flag": 0,
            "candidate_kept_mass": 0.5,
            "candidate_tail_mass": 0.5,
        })
        self.assertAlmostEqual(controller.kept_mass_ema, 0.9)
        controller.update({
            **common,
            "flag": 1,
            "candidate_kept_mass": 0.0,
            "candidate_tail_mass": 1.0,
        })
        self.assertAlmostEqual(controller.kept_mass_ema, 0.9)

    def test_parameterized_predictors_preserve_default_formulas(self):
        controller = core.DynamicDSSDController(controller_args())
        difficulty = 0.3
        expected_alpha = (
            0.65 * controller.alpha_lcb()
            + 0.20 * (1.0 - difficulty)
            + 0.15 * (1.0 - controller.early_reject_ema)
        )
        self.assertAlmostEqual(controller.predict_alpha(difficulty, 0.0), expected_alpha)

        expected_compression = (
            0.45 * controller.kept_mass_ema
            + 0.25 * controller.saving_ema
            + 0.20 * 1.0
            + 0.10 * (1.0 - difficulty)
            - 0.15 * 0.75
        )
        self.assertAlmostEqual(
            controller.predict_compression_success(difficulty, 64, 0.995),
            expected_compression,
        )

    def test_predictor_weights_are_normalized(self):
        controller = core.DynamicDSSDController(controller_args(
            dynamic_alpha_weight_history=2.0,
            dynamic_alpha_weight_difficulty=1.0,
            dynamic_alpha_weight_early_reject=1.0,
            dynamic_cmp_weight_mass=2.0,
            dynamic_cmp_weight_saving=1.0,
            dynamic_cmp_weight_topk=1.0,
            dynamic_cmp_weight_easiness=0.0,
        ))
        self.assertAlmostEqual(controller.alpha_weight_history, 0.5)
        self.assertAlmostEqual(controller.alpha_weight_difficulty, 0.25)
        self.assertAlmostEqual(controller.alpha_weight_early_reject, 0.25)
        self.assertAlmostEqual(controller.cmp_weight_mass, 0.5)
        self.assertAlmostEqual(controller.cmp_weight_saving, 0.25)
        self.assertAlmostEqual(controller.cmp_weight_topk, 0.25)


class RejectPacketTest(unittest.TestCase):
    def test_full_fallback_preserves_sparse_candidate_mass(self):
        packet_args = SimpleNamespace(
            compress_reject_packet=True,
            reject_packet_dtype="float16",
            reject_topk=1,
            reject_mass_threshold=0.9,
            reject_min_prob=0.0,
        )
        packet = core.build_reject_packet(
            torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32), packet_args
        )
        self.assertEqual(packet["mode"], "full")
        self.assertAlmostEqual(packet["kept_mass"], 1.0)
        self.assertAlmostEqual(packet["candidate_kept_mass"], 0.6, places=6)
        self.assertAlmostEqual(packet["candidate_tail_mass"], 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
