"""Reproducible external baselines for the Async-SD evaluation.

The implementations share the existing model, sampling, link-delay, and timing
interfaces. They intentionally do not call the Async-SD controller or shadow path.
"""

import copy
import concurrent.futures
import time
from typing import Callable, Dict, List

import torch
from tqdm import tqdm

import dynamic_dssd_main as core
from pipesd_utils import pipesd_dynamic_token_schedule, pipesd_should_verify


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _latency_fields(stats: Dict, histories: Dict[str, List[float]]) -> None:
    for name, values in histories.items():
        stats.update(core._latency_stats(values, f"round_{name}_latency"))
        stats.update(core._latency_samples(values, f"round_{name}_latency"))


def _dynamic_lookahead_draft(uav, prefix: torch.Tensor, args):
    """Draft until confidence drops below the public dynamic-lookahead threshold."""
    max_gamma = max(1, int(args.lookahead_max_gamma))
    min_gamma = max(1, min(int(args.lookahead_min_gamma), max_gamma))
    threshold = float(args.lookahead_confidence_threshold)
    x = prefix.to(uav.device)
    q_values, q_probs = [], []

    with torch.no_grad():
        for _ in range(max_gamma):
            logits = uav.model(x).logits[:, -1, :]
            q_dist = core.sampling_probs(
                logits, args.temperature, args.top_k, args.top_p
            )[0].cpu()
            token = core.sample(
                logits, args.temperature, args.top_k, args.top_p,
                generator=uav.main_rng,
            )
            token_id = int(token.item())
            q_values.append(float(q_dist[token_id].item()))
            q_probs.append(q_dist)
            x = torch.cat((x, token), dim=1)
            if len(q_values) >= min_gamma and q_values[-1] < threshold:
                break

    gamma = len(q_values)
    uplink_bytes = 16 + gamma * 4 + gamma * 4
    return x, q_values, torch.stack(q_probs), uplink_bytes, gamma


def generate_confidence_lookahead_dssd(uav, bs, input_ids, tokenizer, args):
    """Public confidence-threshold lookahead over the Vanilla DSSD protocol.

    This is the deployed threshold heuristic, not the trained DISCO classifier.
    """
    args.compress_reject_packet = False
    uav.args.compress_reject_packet = False
    bs.args.compress_reject_packet = False
    return _generate_variable_dssd(
        uav, bs, input_ids, tokenizer, args,
        draft_fn=lambda prefix, remain: _dynamic_lookahead_draft(
            uav, prefix, _with_max_gamma(args, remain)
        ),
        run_tag="Confidence Lookahead DSSD",
        extra_stats={
            "lookahead_confidence_threshold": args.lookahead_confidence_threshold,
            "lookahead_min_gamma": args.lookahead_min_gamma,
            "lookahead_max_gamma": args.lookahead_max_gamma,
        },
    )


def _with_max_gamma(args, remain: int):
    """Return a shallow argument copy with the per-round lookahead cap."""
    local = copy.copy(args)
    local.lookahead_max_gamma = min(max(1, remain), int(args.lookahead_max_gamma))
    local.lookahead_min_gamma = min(
        int(args.lookahead_min_gamma), local.lookahead_max_gamma
    )
    return local


def _generate_variable_dssd(
    uav,
    bs,
    input_ids,
    tokenizer,
    args,
    draft_fn: Callable,
    run_tag: str,
    extra_stats: Dict,
):
    input_ids = input_ids.to(uav.device)
    max_total_len = input_ids.shape[1] + int(args.max_len)
    link = core.TimeVaryingLinkReplay.from_args(args)
    prefix = input_ids
    torch.manual_seed(int(args.seed))

    total_comm = total_slm = total_llm = 0.0
    total_uplink_bytes = total_proposed = accepted_total = rounds = 0
    accept_all_rounds = reject_rounds = 0
    reject_full_cnt = 0
    reject_down_actual = reject_down_full = 0
    gamma_hist: List[int] = []
    histories = {"total": [], "comm": [], "slm": [], "llm": []}

    with tqdm(total=max_total_len, desc=run_tag) as bar:
        bar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        start = time.time()
        while prefix.shape[1] < max_total_len:
            round_start = time.time()
            old_len = prefix.shape[1]
            remain = max_total_len - old_len

            slm_start = time.time()
            x_draft, q_values, q_probs, uplink_bytes, gamma = draft_fn(prefix, remain)
            round_slm = time.time() - slm_start
            gamma = int(gamma)
            gamma_hist.append(gamma)
            rounds += 1
            total_proposed += gamma
            total_uplink_bytes += int(uplink_bytes)
            total_slm += round_slm

            t_up = link.delay_bytes(int(uplink_bytes), direction="up")
            total_comm += t_up
            time.sleep(t_up)

            llm_start = time.time()
            j, flag, reject_packet, xj, accepted, _ = bs.verify_DSSD(
                x_draft, q_values, gamma
            )
            round_llm = time.time() - llm_start
            total_llm += round_llm
            accepted_total += int(accepted)
            prefix_len = prefix.shape[1]

            if flag == 1:
                accept_all_rounds += 1
                new_prefix = torch.cat([x_draft, xj.to(uav.device)], dim=1)
                down_bytes = 4 + int(xj.numel()) * 4
            else:
                reject_rounds += 1
                reject_full_cnt += 1
                actual = core.reject_packet_nbytes(reject_packet)
                full = core.full_packet_equivalent_nbytes(reject_packet)
                reject_down_actual += actual
                reject_down_full += full
                correction = uav.resample_DSSD(j, reject_packet, q_probs)
                x_draft[:, prefix_len + j - 1] = correction.to(x_draft.device)
                new_prefix = torch.cat(
                    [prefix, x_draft[:, prefix_len:prefix_len + j]], dim=1
                )
                down_bytes = 4 + actual

            new_prefix = new_prefix[:, :max_total_len]
            t_down = link.delay_bytes(int(down_bytes), direction="down")
            total_comm += t_down
            time.sleep(t_down)
            prefix = new_prefix
            bar.update(prefix.shape[1] - old_len)

            histories["total"].append(time.time() - round_start)
            histories["comm"].append(t_up + t_down)
            histories["slm"].append(round_slm)
            histories["llm"].append(round_llm)

    total_time = time.time() - start
    generated_tokens = prefix.shape[1] - initial_len
    throughput = generated_tokens / max(total_time, 1e-12)
    acceptance = accepted_total / max(total_proposed, 1)
    switches = sum(a != b for a, b in zip(gamma_hist, gamma_hist[1:]))
    stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "reject_compressed_cnt": 0,
        "reject_full_cnt": reject_full_cnt,
        "compression_hit_rate": 0.0,
        "reject_down_bytes_actual": reject_down_actual,
        "reject_down_bytes_full_equiv": reject_down_full,
        "reject_downlink_saving": 0.0,
        "avg_kept_mass": 1.0 if reject_rounds else 0.0,
        "avg_tail_mass": 0.0,
        "avg_gamma": sum(gamma_hist) / max(len(gamma_hist), 1),
        "gamma_usage": core._hist_counts(gamma_hist),
        "compression_on_ratio": 0.0,
        "controller_switch_count": switches,
        "controller_switch_rate": switches / max(rounds, 1),
        "controller_safe_mode_count": 0,
        "controller_safe_mode_rate": 0.0,
        **extra_stats,
    }
    _latency_fields(stats, histories)
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    return (
        generated, throughput, total_time, acceptance, total_comm,
        total_uplink_bytes, rounds, total_slm, total_llm, stats,
    )


def _parse_gamma_set(raw: str) -> List[int]:
    values = sorted({max(1, int(x.strip())) for x in str(raw).split(",") if x.strip()})
    if not values:
        raise ValueError("TK-SLT gamma set must not be empty")
    return values


def _odld_choice(alpha: float, b: float, c: float, gamma_set: List[int]):
    """Discrete ODLD objective from the TK-SLT paper, plus the AS2 decision."""
    alpha = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
    cost = max(0.0, float(b) + float(c))

    def speedup(gamma: int) -> float:
        expected = (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
        return expected / max(1.0 + gamma * cost, 1e-12)

    scored = [(speedup(gamma), gamma) for gamma in gamma_set]
    best_speedup, best_gamma = max(scored)
    return int(best_gamma), float(best_speedup), best_speedup > 1.0


def _tkslt_draft(uav, prefix, gamma: int, args):
    """Top-K proposal and sparse uplink packet used by TK-SLT."""
    x = prefix.to(uav.device)
    vocab_size = int(uav.model.config.vocab_size)
    k = min(max(1, int(args.tkslt_k)), vocab_size)
    packet_dtype = _dtype(args.tkslt_packet_dtype)
    prob_bytes = torch.empty((), dtype=packet_dtype).element_size()
    q_rows = []

    with torch.no_grad():
        for _ in range(gamma):
            logits = uav.model(x).logits[:, -1, :]
            full_q = core.sampling_probs(logits, args.temperature, 0, 0.0)[0]
            top_values, top_ids = torch.topk(full_q, k=k)
            transmitted = top_values.to(packet_dtype).to(torch.float32)
            transmitted = transmitted / transmitted.sum().clamp(min=1e-12)
            sparse_q = torch.zeros_like(full_q, dtype=torch.float32)
            sparse_q[top_ids] = transmitted
            token = torch.multinomial(
                sparse_q, num_samples=1, generator=uav.main_rng
            ).view(1, 1)
            q_rows.append(sparse_q.cpu())
            x = torch.cat((x, token), dim=1)

    per_candidate = 4 + k * 4 + k * prob_bytes
    uplink_bytes = 16 + gamma * per_candidate
    return x, torch.stack(q_rows), uplink_bytes


def _verify_tkslt(bs, x_draft, q_probs: torch.Tensor, gamma: int, args):
    prefix_len = x_draft.shape[1] - gamma
    with torch.no_grad():
        logits = bs.model(x_draft.to(bs.device)).logits.cpu()
    p_probs = core.sampling_probs(
        logits[0, prefix_len - 1:prefix_len + gamma, :],
        args.temperature, args.top_k, args.top_p,
    )
    accepted = 0
    for i in range(gamma):
        token_id = int(x_draft[0, prefix_len + i].item())
        q_value = max(float(q_probs[i, token_id].item()), 1e-12)
        p_value = float(p_probs[i, token_id].item())
        if torch.rand(1).item() > min(1.0, p_value / q_value):
            residual = (p_probs[i] - q_probs[i]).clamp(min=0)
            if residual.sum() <= 1e-12:
                residual = p_probs[i]
            correction = torch.multinomial(residual / residual.sum(), 1).view(1, 1)
            return i + 1, False, correction, accepted
        accepted += 1
    continuation = torch.multinomial(p_probs[gamma], 1).view(1, 1)
    return gamma + 1, True, continuation, accepted


def _target_only_round(bs, prefix: torch.Tensor, args):
    with torch.no_grad():
        logits = bs.model(prefix.to(bs.device)).logits[:, -1, :]
        token = core.sample(logits, args.temperature, args.top_k, args.top_p)
    if torch.device(bs.device).type == "cuda":
        torch.cuda.synchronize(torch.device(bs.device))
    # Materialize the edge result before downlink delay is simulated; otherwise
    # asynchronous CUDA work can incorrectly overlap target inference and sleep.
    return token.detach().cpu()


def generate_tkslt_as2(uav, bs, input_ids, tokenizer, args):
    """TK-SLT with online ODLD estimates and the paper's AS2 fallback decision."""
    input_ids = input_ids.to(uav.device)
    max_total_len = input_ids.shape[1] + int(args.max_len)
    link = core.TimeVaryingLinkReplay.from_args(args)
    prefix = input_ids
    torch.manual_seed(int(args.seed))
    gamma_set = _parse_gamma_set(args.tkslt_gamma_set)
    beta = float(args.tkslt_beta)
    warmup_rounds = max(1, int(args.tkslt_warmup_rounds))
    alpha_ema = float(args.tkslt_initial_alpha)
    slm_per_token_ema = float(args.tkslt_initial_slm_ms) / 1000.0
    llm_ema = float(args.tkslt_initial_llm_ms) / 1000.0
    k = min(int(args.tkslt_k), int(uav.model.config.vocab_size))
    prob_bytes = torch.empty((), dtype=_dtype(args.tkslt_packet_dtype)).element_size()

    total_comm = total_slm = total_llm = 0.0
    total_uplink = total_proposed = accepted_total = rounds = 0
    accept_all_rounds = reject_rounds = fallback_rounds = 0
    gamma_hist: List[int] = []
    predicted_speedups: List[float] = []
    histories = {"total": [], "comm": [], "slm": [], "llm": []}

    with tqdm(total=max_total_len, desc="TK-SLT + ODLD/AS2") as bar:
        bar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        start = time.time()
        while prefix.shape[1] < max_total_len:
            round_start = time.time()
            old_len = prefix.shape[1]
            rounds += 1
            remain = max_total_len - old_len
            candidates = [g for g in gamma_set if g <= remain] or [1]
            serialization_per_candidate = (4 + k * (4 + prob_bytes)) / max(
                float(args.bandwidth) * 1e6 / 8.0, 1e-12
            )
            b = serialization_per_candidate / max(llm_ema, 1e-9)
            c = slm_per_token_ema / max(llm_ema, 1e-9)
            gamma, predicted_speedup, use_dsd = _odld_choice(
                alpha_ema, b, c, candidates
            )
            if rounds <= warmup_rounds:
                gamma = min(max(2, min(candidates)), max(candidates))
                use_dsd = True
            predicted_speedups.append(predicted_speedup)

            if not use_dsd:
                gamma_hist.append(0)
                fallback_rounds += 1
                request_bytes = 16
                t_up = link.delay_bytes(request_bytes, direction="up")
                total_uplink += request_bytes
                total_comm += t_up
                time.sleep(t_up)
                llm_start = time.time()
                token = _target_only_round(bs, prefix, args)
                round_llm = time.time() - llm_start
                round_slm = 0.0
                total_llm += round_llm
                t_down = link.delay_bytes(4, direction="down")
                total_comm += t_down
                time.sleep(t_down)
                prefix = torch.cat([prefix, token.to(uav.device)], dim=1)
            else:
                gamma = min(gamma, remain)
                gamma_hist.append(gamma)
                total_proposed += gamma
                slm_start = time.time()
                x_draft, q_probs, uplink_bytes = _tkslt_draft(
                    uav, prefix, gamma, args
                )
                round_slm = time.time() - slm_start
                total_slm += round_slm
                total_uplink += uplink_bytes
                t_up = link.delay_bytes(uplink_bytes, direction="up")
                total_comm += t_up
                time.sleep(t_up)

                llm_start = time.time()
                j, accept_all, correction, accepted = _verify_tkslt(
                    bs, x_draft, q_probs, gamma, args
                )
                round_llm = time.time() - llm_start
                total_llm += round_llm
                accepted_total += accepted
                if accept_all:
                    accept_all_rounds += 1
                    new_prefix = torch.cat(
                        [x_draft, correction.to(uav.device)], dim=1
                    )
                else:
                    reject_rounds += 1
                    new_prefix = torch.cat(
                        [prefix, x_draft[:, old_len:old_len + j - 1],
                         correction.to(uav.device)], dim=1
                    )
                prefix = new_prefix[:, :max_total_len]
                t_down = link.delay_bytes(8, direction="down")
                total_comm += t_down
                time.sleep(t_down)

                # Only proposals up to the first rejection are verified. Treating
                # the unverified suffix as rejected biases the online alpha estimate.
                verified = accepted + (0 if accept_all else 1)
                round_alpha = accepted / max(verified, 1)
                alpha_ema = (1.0 - beta) * alpha_ema + beta * round_alpha
                slm_per_token_ema = (
                    (1.0 - beta) * slm_per_token_ema
                    + beta * round_slm / max(gamma, 1)
                )
                llm_ema = (1.0 - beta) * llm_ema + beta * round_llm

            bar.update(prefix.shape[1] - old_len)
            histories["total"].append(time.time() - round_start)
            histories["comm"].append(t_up + t_down)
            histories["slm"].append(round_slm)
            histories["llm"].append(round_llm)

    total_time = time.time() - start
    generated_tokens = prefix.shape[1] - initial_len
    throughput = generated_tokens / max(total_time, 1e-12)
    acceptance = accepted_total / max(total_proposed, 1)
    switches = sum(a != b for a, b in zip(gamma_hist, gamma_hist[1:]))
    stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "reject_compressed_cnt": 0,
        "reject_full_cnt": 0,
        "compression_hit_rate": 0.0,
        "reject_down_bytes_actual": 0,
        "reject_down_bytes_full_equiv": 0,
        "reject_downlink_saving": 0.0,
        "avg_kept_mass": 0.0,
        "avg_tail_mass": 0.0,
        "avg_gamma": total_proposed / max(rounds - fallback_rounds, 1),
        "gamma_usage": core._hist_counts(gamma_hist),
        # TK-SLT sparsifies the draft-side uplink, not Async-SD's reject downlink.
        "compression_on_ratio": 0.0,
        "controller_switch_count": switches,
        "controller_switch_rate": switches / max(rounds, 1),
        "controller_safe_mode_count": fallback_rounds,
        "controller_safe_mode_rate": fallback_rounds / max(rounds, 1),
        "tkslt_k": k,
        "tkslt_uplink_sparse_ratio": (
            (rounds - fallback_rounds) / max(rounds, 1)
        ),
        "tkslt_alpha_final": alpha_ema,
        "tkslt_predicted_speedup_mean": sum(predicted_speedups) / max(len(predicted_speedups), 1),
        "tkslt_as2_fallback_rounds": fallback_rounds,
        "tkslt_as2_fallback_rate": fallback_rounds / max(rounds, 1),
    }
    _latency_fields(stats, histories)
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    return (
        generated, throughput, total_time, acceptance, total_comm, total_uplink,
        rounds, total_slm, total_llm, stats,
    )


def _pipesd_draft_block(
    uav, prefix: torch.Tensor, args, generator: torch.Generator,
    max_gamma: int, stop_when: Callable = None,
):
    """Draft one hybrid-triggered PipeSD verification block."""
    x = prefix.to(uav.device)
    q_rows: List[torch.Tensor] = []
    confidences: List[float] = []
    token_times: List[float] = []
    rng_states = []
    min_gamma = max(1, int(args.pipesd_min_gamma))
    max_gamma = max(1, min(int(args.pipesd_max_gamma), int(max_gamma)))

    with torch.no_grad():
        for _ in range(max_gamma):
            if stop_when is not None and q_rows and stop_when():
                break
            token_start = time.time()
            logits = uav.model(x).logits[:, -1, :]
            q_dist = core.sampling_probs(
                logits, args.temperature, args.top_k, args.top_p
            )[0]
            token = core.sample(
                logits, args.temperature, args.top_k, args.top_p,
                generator=generator,
            )
            x = torch.cat((x, token), dim=1)
            q_rows.append(q_dist.detach().to(torch.float32).cpu())
            confidences.append(float(q_dist.max().item()))
            rng_states.append(generator.get_state().clone())
            token_times.append(time.time() - token_start)
            if len(q_rows) >= min_gamma and pipesd_should_verify(
                confidences,
                args.pipesd_single_threshold,
                args.pipesd_multi_threshold,
            ):
                break

    return {
        "x_draft": x.detach(),
        "q_probs": torch.stack(q_rows),
        "gamma": len(q_rows),
        "confidences": confidences,
        "token_compute_times": token_times,
        "rng_states": rng_states,
        "rng_state_final": generator.get_state().clone(),
    }


def _pipesd_uplink_plan(block: Dict, link, args) -> Dict:
    prob_bytes = torch.empty((), dtype=_dtype(args.pipesd_packet_dtype)).element_size()
    vocab_size = int(block["q_probs"].shape[-1])
    per_token_bytes = 4 + vocab_size * prob_bytes
    fixed_cost = link.delay_bytes(16, direction="up")
    per_token_cost = max(
        0.0,
        link.delay_bytes(16 + per_token_bytes, direction="up") - fixed_cost,
    )
    batches, completion = pipesd_dynamic_token_schedule(
        block["token_compute_times"], fixed_cost, per_token_cost
    )
    uplink_bytes = len(batches) * 16 + len(block["token_compute_times"]) * per_token_bytes
    service = sum(
        link.delay_bytes(16 + len(batch) * per_token_bytes, direction="up")
        for batch in batches
    )
    return {
        "batches": batches,
        "completion_s": completion,
        "uplink_bytes": uplink_bytes,
        "uplink_service_s": service,
        "per_token_bytes": per_token_bytes,
    }


def _verify_pipesd(bs, x_draft, q_probs: torch.Tensor, gamma: int, args):
    """Target-side exact speculative verification with PipeSD's uploaded Q rows."""
    prefix_len = x_draft.shape[1] - gamma
    with torch.no_grad():
        logits = bs.model(x_draft.to(bs.device)).logits
    p_probs = core.sampling_probs(
        logits[0, prefix_len - 1:prefix_len + gamma, :],
        args.temperature, args.top_k, args.top_p,
    )
    accepted = 0
    for index in range(gamma):
        token_id = int(x_draft[0, prefix_len + index].item())
        q_row = q_probs[index].to(bs.device)
        q_value = max(float(q_row[token_id].item()), 1e-12)
        p_value = float(p_probs[index, token_id].item())
        if torch.rand(1).item() > min(1.0, p_value / q_value):
            residual = (p_probs[index] - q_row).clamp(min=0)
            if float(residual.sum().item()) <= 1e-12:
                residual = p_probs[index]
            correction = torch.multinomial(residual / residual.sum(), 1).view(1, 1)
            return index + 1, False, correction.detach().cpu(), accepted
        accepted += 1
    continuation = torch.multinomial(p_probs[gamma], 1).view(1, 1)
    return gamma + 1, True, continuation.detach().cpu(), accepted


def _pipesd_remote_round(bs, block: Dict, uplink_wait_s: float, link, args):
    if uplink_wait_s > 0.0:
        time.sleep(uplink_wait_s)
    llm_start = time.time()
    j, accept_all, correction, accepted = _verify_pipesd(
        bs, block["x_draft"], block["q_probs"], block["gamma"], args
    )
    llm_time = time.time() - llm_start
    downlink_s = link.delay_bytes(8, direction="down")
    time.sleep(downlink_s)
    return {
        "j": j,
        "accept_all": accept_all,
        "correction": correction,
        "accepted": accepted,
        "llm_time_s": llm_time,
        "downlink_s": downlink_s,
    }


def _pipesd_shadow_block(uav, current_block: Dict, max_gamma: int, args, remote_future):
    """Generate PipeSD's optimistic anchor and waiting block while verification runs."""
    if max_gamma < 1 or remote_future.done():
        return None
    branch_rng = torch.Generator(device=uav.rng_device)
    branch_rng.set_state(uav.main_rng.get_state())
    with torch.no_grad():
        logits = uav.model(current_block["x_draft"].to(uav.device)).logits[:, -1, :]
        q_dist = core.sampling_probs(
            logits, args.temperature, args.top_k, args.top_p
        )[0]
        anchor = core.sample(
            logits, args.temperature, args.top_k, args.top_p,
            generator=branch_rng,
        )
    if remote_future.done():
        return None
    optimistic_prefix = torch.cat(
        [current_block["x_draft"].to(uav.device), anchor], dim=1
    )
    block = _pipesd_draft_block(
        uav, optimistic_prefix, args, branch_rng, max_gamma,
        stop_when=remote_future.done,
    )
    block["anchor_token_id"] = int(anchor.item())
    block["optimistic_prefix"] = optimistic_prefix.detach()
    return block


def generate_pipesd(uav, bs, input_ids, tokenizer, args):
    """Reproduce PipeSD's hybrid trigger, DP upload batching, and waiting pipeline.

    The public PipeSD control rules are retained while model execution and wireless
    delay use this repository's common CPU/GPU and link interfaces.
    """
    input_ids = input_ids.to(uav.device)
    max_total_len = input_ids.shape[1] + int(args.max_len)
    link = core.TimeVaryingLinkReplay.from_args(args)
    prefix = input_ids
    torch.manual_seed(int(args.seed))

    total_comm = total_slm = total_llm = 0.0
    total_uplink = total_proposed = accepted_total = rounds = 0
    accept_all_rounds = reject_rounds = 0
    gamma_hist: List[int] = []
    histories = {"total": [], "comm": [], "slm": [], "llm": []}
    trigger_single = trigger_multi = trigger_both = 0
    dp_batch_total = uplink_service_total = uplink_hidden_total = 0.0
    waiting_launches = waiting_hits = 0
    waiting_discard_reject = waiting_discard_mismatch = 0
    waiting_tokens = rollback_tokens = 0
    waiting_compute_total = waiting_overhang_total = 0.0
    cached = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with tqdm(total=max_total_len, desc="PipeSD") as bar:
            bar.update(prefix.shape[1])
            initial_len = prefix.shape[1]
            start = time.time()
            while prefix.shape[1] < max_total_len:
                round_start = time.time()
                old_len = prefix.shape[1]
                remain = max_total_len - old_len
                rounds += 1

                if cached is None:
                    draft_start = time.time()
                    block = _pipesd_draft_block(
                        uav, prefix, args, uav.main_rng, remain
                    )
                    round_slm = time.time() - draft_start
                    plan = _pipesd_uplink_plan(block, link, args)
                    uplink_wait = max(0.0, plan["completion_s"] - round_slm)
                    hidden_uplink = max(0.0, plan["uplink_service_s"] - uplink_wait)
                    plan_preaccounted = False
                else:
                    block = cached["block"]
                    plan = cached["plan"]
                    gamma = min(int(block["gamma"]), remain)
                    if gamma < int(block["gamma"]):
                        prefix_len = block["optimistic_prefix"].shape[1]
                        block = dict(block)
                        block["gamma"] = gamma
                        block["x_draft"] = block["x_draft"][:, :prefix_len + gamma]
                        block["q_probs"] = block["q_probs"][:gamma]
                        block["confidences"] = block["confidences"][:gamma]
                        block["token_compute_times"] = block["token_compute_times"][:gamma]
                        block["rng_state_final"] = block["rng_states"][gamma - 1]
                        plan = _pipesd_uplink_plan(block, link, args)
                    uav.main_rng.set_state(block["rng_state_final"])
                    round_slm = 0.0
                    elapsed = max(0.0, time.time() - cached["started_at"])
                    uplink_wait = max(0.0, plan["completion_s"] - elapsed)
                    hidden_uplink = max(0.0, plan["uplink_service_s"] - uplink_wait)
                    plan_preaccounted = bool(cached.get("plan_preaccounted", False))
                    cached = None

                gamma = int(block["gamma"])
                total_slm += round_slm
                total_proposed += gamma
                gamma_hist.append(gamma)
                if not plan_preaccounted:
                    total_uplink += int(plan["uplink_bytes"])
                    dp_batch_total += len(plan["batches"])
                    uplink_service_total += float(plan["uplink_service_s"])
                    uplink_hidden_total += hidden_uplink

                product = 1.0
                for confidence in block["confidences"]:
                    product *= confidence
                single = block["confidences"][-1] < float(args.pipesd_single_threshold)
                multi = product < float(args.pipesd_multi_threshold)
                trigger_single += int(single)
                trigger_multi += int(multi)
                trigger_both += int(single and multi)

                remote_started = time.time()
                remote_future = executor.submit(
                    _pipesd_remote_round, bs, block, uplink_wait, link, args
                )
                optimistic_remaining = max_total_len - (old_len + gamma + 1)
                shadow_started = time.time()
                shadow = _pipesd_shadow_block(
                    uav, block, optimistic_remaining, args, remote_future
                )
                shadow_compute = time.time() - shadow_started if shadow is not None else 0.0
                waiting_compute_total += shadow_compute
                if shadow is not None and shadow["gamma"] > 0:
                    waiting_launches += 1
                    waiting_tokens += int(shadow["gamma"])
                    shadow_plan = _pipesd_uplink_plan(shadow, link, args)
                    # PipeSD sends waiting proposals while the current request is
                    # in flight, so count their traffic even if rollback follows.
                    total_uplink += int(shadow_plan["uplink_bytes"])
                    dp_batch_total += len(shadow_plan["batches"])
                    uplink_service_total += float(shadow_plan["uplink_service_s"])
                    shadow_elapsed = max(0.0, time.time() - shadow_started)
                    shadow_remaining = max(
                        0.0, shadow_plan["completion_s"] - shadow_elapsed
                    )
                    uplink_hidden_total += max(
                        0.0, shadow_plan["uplink_service_s"] - shadow_remaining
                    )
                else:
                    shadow = None
                    shadow_plan = None

                remote = remote_future.result()
                remote_elapsed = time.time() - remote_started
                waiting_overhang_total += max(0.0, shadow_compute - remote_elapsed)
                total_llm += remote["llm_time_s"]
                critical_comm = uplink_wait + remote["downlink_s"]
                total_comm += critical_comm
                accepted_total += int(remote["accepted"])

                if remote["accept_all"]:
                    accept_all_rounds += 1
                    new_prefix = torch.cat(
                        [block["x_draft"], remote["correction"].to(uav.device)], dim=1
                    )
                else:
                    reject_rounds += 1
                    j = int(remote["j"])
                    new_prefix = torch.cat(
                        [prefix, block["x_draft"][:, old_len:old_len + j - 1],
                         remote["correction"].to(uav.device)], dim=1
                    )
                new_prefix = new_prefix[:, :max_total_len]

                if shadow is not None:
                    anchor_match = (
                        remote["accept_all"]
                        and int(remote["correction"].item()) == int(shadow["anchor_token_id"])
                    )
                    if anchor_match and new_prefix.shape[1] < max_total_len:
                        waiting_hits += 1
                        cached = {
                            "block": shadow,
                            "plan": shadow_plan,
                            "started_at": shadow_started,
                            "plan_preaccounted": True,
                        }
                    else:
                        rollback_tokens += int(shadow["gamma"])
                        if not remote["accept_all"]:
                            waiting_discard_reject += 1
                        else:
                            waiting_discard_mismatch += 1

                prefix = new_prefix
                bar.update(prefix.shape[1] - old_len)
                histories["total"].append(time.time() - round_start)
                histories["comm"].append(critical_comm)
                histories["slm"].append(round_slm)
                histories["llm"].append(remote["llm_time_s"])

    total_time = time.time() - start
    generated_tokens = prefix.shape[1] - initial_len
    throughput = generated_tokens / max(total_time, 1e-12)
    acceptance = accepted_total / max(total_proposed, 1)
    switches = sum(a != b for a, b in zip(gamma_hist, gamma_hist[1:]))
    stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "avg_gamma": sum(gamma_hist) / max(len(gamma_hist), 1),
        "gamma_usage": core._hist_counts(gamma_hist),
        "compression_on_ratio": 0.0,
        "controller_switch_count": switches,
        "controller_switch_rate": switches / max(rounds, 1),
        "controller_safe_mode_count": 0,
        "controller_safe_mode_rate": 0.0,
        "pipesd_single_threshold": float(args.pipesd_single_threshold),
        "pipesd_multi_threshold": float(args.pipesd_multi_threshold),
        "pipesd_min_gamma": int(args.pipesd_min_gamma),
        "pipesd_max_gamma": int(args.pipesd_max_gamma),
        "pipesd_trigger_single_count": trigger_single,
        "pipesd_trigger_multi_count": trigger_multi,
        "pipesd_trigger_both_count": trigger_both,
        "pipesd_dp_batch_count": int(dp_batch_total),
        "pipesd_dp_batches_per_round": dp_batch_total / max(rounds, 1),
        "pipesd_uplink_service_time_s": uplink_service_total,
        "pipesd_uplink_hidden_time_s": uplink_hidden_total,
        "pipesd_waiting_launches": waiting_launches,
        "pipesd_waiting_hits": waiting_hits,
        "pipesd_waiting_reuse_rate": waiting_hits / max(waiting_launches, 1),
        "pipesd_waiting_discard_reject": waiting_discard_reject,
        "pipesd_waiting_discard_mismatch": waiting_discard_mismatch,
        "pipesd_waiting_tokens": waiting_tokens,
        "pipesd_rollback_tokens": rollback_tokens,
        "pipesd_waiting_compute_time_s": waiting_compute_total,
        "pipesd_waiting_overhang_time_s": waiting_overhang_total,
    }
    _latency_fields(stats, histories)
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    return (
        generated, throughput, total_time, acceptance, total_comm, total_uplink,
        rounds, total_slm, total_llm, stats,
    )
