import argparse
from speculative import (
    Recorder,
    compress_logits,
    normal_generate,
    sample,
    sampling_probs,
    tensor_nbytes,
    tx_delay_bytes,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Dict
import math
import time
from tqdm import tqdm
import json
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
from datetime import datetime


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _latency_stats(values: List[float], prefix: str) -> Dict:
    values_ms = [float(v) * 1000.0 for v in values]
    return {
        f"{prefix}_mean_ms": sum(values_ms) / max(len(values_ms), 1),
        f"{prefix}_p50_ms": _percentile(values_ms, 50),
        f"{prefix}_p95_ms": _percentile(values_ms, 95),
        f"{prefix}_p99_ms": _percentile(values_ms, 99),
    }


def _latency_samples(values: List[float], prefix: str) -> Dict:
    return {f"{prefix}_samples_ms": [float(v) * 1000.0 for v in values]}


def _hist_counts(values: List[int]) -> Dict[int, int]:
    return {int(v): values.count(v) for v in sorted(set(values))}


def _json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True)


def _short_input_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


class TimeVaryingLinkReplay:
    """Step-wise replay of time-varying uplink/downlink bandwidth and RTT."""

    def __init__(
        self,
        entries: List[Dict],
        default_rtt_s: float,
        default_bw_mbps: float,
        speedup: float = 1.0,
    ):
        self.entries = sorted(entries, key=lambda x: float(x.get("time_s", 0.0)))
        self.default_rtt_s = float(default_rtt_s)
        self.default_bw_mbps = float(default_bw_mbps)
        self.speedup = max(float(speedup), 1e-9)
        self.start_time = time.time()
        self.last_transfer_states = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        trace_path = str(getattr(args, "link_trace_path", "") or "").strip()
        speedup = float(getattr(args, "link_trace_speedup", 1.0))
        if not trace_path:
            return cls([], float(args.rtt), float(args.bandwidth), speedup)
        with open(trace_path, "r", encoding="utf-8") as f:
            if trace_path.endswith(".jsonl"):
                entries = [json.loads(line) for line in f if line.strip()]
            else:
                payload = json.load(f)
                entries = payload.get("trace", payload if isinstance(payload, list) else [])
        return cls(entries, float(args.rtt), float(args.bandwidth), speedup)

    def _current_entry(self) -> Dict:
        if not self.entries:
            return {}
        elapsed = (time.time() - self.start_time) * self.speedup
        current = self.entries[0]
        for entry in self.entries:
            if float(entry.get("time_s", 0.0)) <= elapsed:
                current = entry
            else:
                break
        return current

    def current_state(self) -> Dict[str, float]:
        entry = self._current_entry()
        rtt_ms = float(entry.get("rtt_ms", self.default_rtt_s * 1e3))
        return {
            "bw_up_mbps": float(
                entry.get("bw_up_mbps", entry.get("bw_mbps", self.default_bw_mbps))
            ),
            "bw_down_mbps": float(
                entry.get("bw_down_mbps", entry.get("bw_mbps", self.default_bw_mbps))
            ),
            "rtt_ms": rtt_ms,
        }

    @staticmethod
    def state_delay_bytes(state: Dict[str, float], size_B: int, direction: str) -> float:
        key = "bw_up_mbps" if direction == "up" else "bw_down_mbps"
        bw_mbps = float(state[key])
        bw_Bps = max(1e-6, bw_mbps * 1e6 / 8.0)
        return (float(state["rtt_ms"]) / 1000.0) / 2.0 + float(size_B) / bw_Bps

    def delay_bytes(self, size_B: int, direction: str = "down") -> float:
        state = self.current_state()
        self.last_transfer_states[direction] = dict(state)
        return self.state_delay_bytes(state, size_B, direction)

    def completed_round_observation(self) -> Dict[str, float]:
        """Return link telemetry from transfers completed in the latest round."""
        up = self.last_transfer_states.get("up")
        down = self.last_transfer_states.get("down")
        fallback = down or up or self.current_state()
        return {
            "bw_up_mbps": float((up or fallback)["bw_up_mbps"]),
            "bw_down_mbps": float((down or fallback)["bw_down_mbps"]),
            "rtt_ms": float((down or up or fallback)["rtt_ms"]),
        }


class ControllerLinkTelemetry:
    """Expose either instantaneous or completed-round link state to the controller."""

    def __init__(self, replay: TimeVaryingLinkReplay, args: argparse.Namespace):
        self.replay = replay
        self.mode = str(getattr(args, "controller_link_mode", "causal_previous_round"))
        if self.mode not in {"instantaneous", "causal_previous_round"}:
            raise ValueError(f"unsupported controller link mode: {self.mode}")
        # The request-start sample is causal and provides a cold-start estimate.
        self.observed_state = replay.current_state()
        self.observation_count = 0

    def delay_bytes(self, size_B: int, direction: str = "down") -> float:
        state = (
            self.replay.current_state()
            if self.mode == "instantaneous"
            else self.observed_state
        )
        return self.replay.state_delay_bytes(state, size_B, direction)

    def observe_completed_round(self) -> None:
        if self.mode == "causal_previous_round":
            self.observed_state = self.replay.completed_round_observation()
        self.observation_count += 1

    def stats(self) -> Dict[str, object]:
        return {
            "controller_link_mode": self.mode,
            "controller_link_observation_count": self.observation_count,
            "controller_link_bw_up_mbps_final": self.observed_state["bw_up_mbps"],
            "controller_link_bw_down_mbps_final": self.observed_state["bw_down_mbps"],
            "controller_link_rtt_ms_final": self.observed_state["rtt_ms"],
        }



def _reject_packet_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

def build_reject_packet(pj: torch.Tensor, args: argparse.Namespace) -> Dict:
    """
    将拒绝位置的 P_j 分布压缩成 reject packet。
    压缩策略：
      1) 若 top-k 累积质量足够高，则仅发送 top-k id + probs
      2) 否则自动回退为 full packet
    注意：压缩 packet 是近似的；若要求严格 exact，请关闭压缩或把阈值调到 1.0。
    """
    use_compression = getattr(args, "compress_reject_packet", True)
    vocab_size = pj.numel()
    dtype = _reject_packet_dtype(getattr(args, "reject_packet_dtype", "float16"))
    prob_dtype_bytes = _dtype_nbytes(dtype)
    if (not use_compression) or vocab_size <= 0:
        return {
            "mode": "full",
            "probs": pj.to(dtype).cpu(),
            "vocab_size": vocab_size,
            "kept_mass": 1.0,
            "tail_mass": 0.0,
            "candidate_kept_mass": 1.0,
            "candidate_tail_mass": 0.0,
            "prob_dtype_bytes": prob_dtype_bytes,
        }

    topk = min(getattr(args, "reject_topk", 64), vocab_size)
    mass_threshold = getattr(args, "reject_mass_threshold", 0.995)
    min_prob = getattr(args, "reject_min_prob", 0.0)

    top_vals, top_ids = torch.topk(pj, k=topk)
    total_mass = max(float(pj.to(torch.float64).sum().item()), 1e-12)
    kept_mass = min(1.0, max(0.0, float(top_vals.to(torch.float64).sum().item()) / total_mass))
    tail_mass = max(0.0, 1.0 - kept_mass)
    smallest_kept = float(top_vals[-1].item()) if topk > 0 else 0.0

    # 若 top-k 覆盖质量不够，或尾部仍然比较重，则回退发送完整分布
    if kept_mass < mass_threshold or smallest_kept < min_prob:
        return {
            "mode": "full",
            "probs": pj.to(dtype).cpu(),
            "vocab_size": vocab_size,
            "kept_mass": 1.0,
            "tail_mass": 0.0,
            "candidate_kept_mass": kept_mass,
            "candidate_tail_mass": tail_mass,
            "prob_dtype_bytes": prob_dtype_bytes,
        }

    return {
        "mode": "compressed",
        "ids": top_ids.to(torch.int32).cpu(),
        "probs": top_vals.to(dtype).cpu(),
        "vocab_size": vocab_size,
        "kept_mass": kept_mass,
        "tail_mass": tail_mass,
        "candidate_kept_mass": kept_mass,
        "candidate_tail_mass": tail_mass,
        "prob_dtype_bytes": prob_dtype_bytes,
    }

def reconstruct_pj_from_packet(packet: Dict) -> torch.Tensor:
    """
    将 reject packet 还原成一个稀疏/完整的 P_j 向量（CPU 上）。
    """
    if packet["mode"] == "full":
        return packet["probs"].to(torch.float32).cpu()

    vocab_size = int(packet["vocab_size"])
    pj = torch.zeros(vocab_size, dtype=torch.float32)
    ids = packet["ids"].to(torch.long).cpu()
    vals = packet["probs"].to(torch.float32).cpu()
    pj[ids] = vals
    return pj

def reject_packet_nbytes(packet: Dict) -> int:
    """
    统计 reject packet 的字节数，用于通信延迟模拟。
    说明：这里的大小包含少量 packet 元数据，用于更接近真实协议传输开销。
    """
    meta_bytes = 16
    if packet["mode"] == "full":
        return meta_bytes + tensor_nbytes(packet["probs"])
    return meta_bytes + tensor_nbytes(packet["ids"]) + tensor_nbytes(packet["probs"])


def full_packet_equivalent_nbytes(packet: Dict) -> int:
    """
    若该 reject 位置不做稀疏化，以相同概率精度发送完整 P_j 时的等效字节数。
    仅用于统计压缩收益，不参与实际传输。
    """
    meta_bytes = 16
    vocab_size = int(packet["vocab_size"])
    return meta_bytes + vocab_size * int(packet.get("prob_dtype_bytes", 2))


class UAVNode:
    """无人机节点 - 负责draft阶段的小模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        rng_device = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
        self.rng_device = rng_device
        self.main_rng = torch.Generator(device=rng_device)
        self.main_rng.manual_seed(int(getattr(args, 'seed', 1234)))
        self.shadow_rng = torch.Generator(device=rng_device)
        self.shadow_rng.manual_seed(int(getattr(args, 'seed', 1234)) + 1_000_000)
        
    def draft_step(self, prefix, gamma):
        """
        UAV端执行draft步骤，生成gamma个候选token
        返回：
          x_draft: (1, prefix_len+γ) 
          q_step_logits_stack: (γ, V)
          dup_bytes: 传输字节数
        """
        x = prefix.to(self.device)
        q_stack = []
        
        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits  # (1, seq, V)
                q_stack.append(logits[0, -1].cpu())  # 存储最后一行 (V,)
                next_tok = sample(logits[:, -1, :],
                                self.args.temperature, 
                                self.args.top_k, 
                                self.args.top_p,
                                generator=self.main_rng)
                x = torch.cat((x, next_tok), dim=1)
        
        q_step_logits = torch.stack(q_stack, dim=0)  # -> (γ, V)
        
        # 计算传输字节数
        raw_bytes = tensor_nbytes(q_step_logits)
        if self.args.use_dist_summary:
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_step_logits)
            dup_bytes = comp_bytes
        else:
            dup_bytes = raw_bytes
            
        return x, q_step_logits, dup_bytes
    
    def draft_DSSD(
        self, prefix: torch.Tensor, gamma: int, generator: torch.Generator = None,
        cancel_event=None, rng_state_history=None, progress_counter=None,
        first_logits: torch.Tensor = None,
    ):
        """
        DSSD Draft阶段：生成γ个推测token，记录概率值（上行传输）和完整分布（本地重新采样）
        返回：
          x_draft: 前缀+推测token（shape: (1, seq_len+gamma)）
          q_values: 每个推测token的概率值（list[float], 长度gamma）→ 上行传输
          q_probs: 每个推测token的完整分布（shape: (gamma, V)）→ 本地保存
          dup_bytes: 上行传输字节数（token+概率值）
        """
        x = prefix.to(self.device)
        q_probs = []  # 保存完整分布（本地用于重新采样）
        q_values = []  # 保存概率值（上行传输）

        with torch.no_grad():
            for step in range(gamma):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                # 1. 小模型前向计算，得到当前分布Q_i(x)
                if step == 0 and first_logits is not None:
                    next_logits = first_logits.to(self.device)
                else:
                    next_logits = self.model(x).logits[:, -1, :]
                if cancel_event is not None and cancel_event.is_set():
                    return None
                q_dist = sampling_probs(
                    next_logits, self.args.temperature,
                    self.args.top_k, self.args.top_p
                )[0].cpu()
                q_probs.append(q_dist)
                
                # 2. 采样下一个token x_i ~ Q_i(x)
                next_tok = sample(next_logits, self.args.temperature, self.args.top_k, self.args.top_p, generator=(generator if generator is not None else self.main_rng))
                x = torch.cat((x, next_tok), dim=1)
                if rng_state_history is not None:
                    active_rng = generator if generator is not None else self.main_rng
                    rng_state_history.append(active_rng.get_state().clone())
                if progress_counter is not None:
                    progress_counter["tokens"] = progress_counter.get("tokens", 0) + 1
                
                # 3. 记录x_i的概率值q_i = Q_i(x_i)（用于上行传输）
                tok_id = next_tok.item()
                q_values.append(q_dist[tok_id].item())  # q_i = P_Q(x_i)

        # 转换为张量（便于后续处理）
        q_probs = torch.stack(q_probs, dim=0)  # (gamma, V)
        
        # 计算上行传输字节数（仅统计本轮新增的 draft token + 概率值 + 少量协议元数据）
        # 注意：真实会话化 DSSD 不应把整个 prefix 都重复计入上行。
        meta_bytes = 16
        token_bytes = gamma * 4  # 每个 draft token 按 int32 计
        prob_bytes = len(q_values) * 4  # 每个概率值按 float32 计
        dup_bytes = meta_bytes + token_bytes + prob_bytes

        return x, q_values, q_probs, dup_bytes

    @torch.no_grad()
    def preview_difficulty(self, prefix: torch.Tensor) -> Dict[str, float]:
        """
        用当前 prefix 做一次轻量 preview，估计当前 token 难度。
        返回归一化 entropy、top1 概率和 top1-top2 margin。
        """
        x = prefix.to(self.device)
        logits = self.model(x).logits[:, -1, :]
        probs = sampling_probs(
            logits, self.args.temperature, self.args.top_k, self.args.top_p
        )[0].detach().cpu()
        result = self.difficulty_from_probs(probs)
        result["first_logits"] = logits.detach()
        return result

    @staticmethod
    def difficulty_from_probs(probs: torch.Tensor) -> Dict[str, float]:
        """Recover the controller's difficulty features from a cached draft distribution."""
        probs = probs.detach().to(torch.float32).cpu()
        vocab = max(2, probs.numel())
        entropy = float((-(probs * (probs.clamp(min=1e-12)).log())).sum().item())
        entropy_norm = entropy / math.log(vocab)
        top2_vals, _ = torch.topk(probs, k=2)
        qmax = float(top2_vals[0].item())
        margin = float((top2_vals[0] - top2_vals[1]).item())
        difficulty = 0.5 * entropy_norm + 0.3 * (1.0 - qmax) + 0.2 * (1.0 - margin)
        difficulty = max(0.0, min(1.0, difficulty))
        return {
            "entropy_norm": entropy_norm,
            "qmax": qmax,
            "margin": margin,
            "difficulty": difficulty,
        }

    @torch.no_grad()
    def sample_next_token_info(self, prefix: torch.Tensor) -> Dict:
        """
        Predict the target continuation used to anchor the next-round shadow.

        ``sample`` preserves the original stochastic anchor. ``map`` uses the
        draft distribution's top-1 token; this changes only speculative work,
        because a shadow is committed only after matching the target token.
        """
        x = prefix.to(self.device)
        logits = self.model(x).logits
        q_dist = sampling_probs(
            logits[:, -1, :], self.args.temperature,
            self.args.top_k, self.args.top_p
        )[0].cpu()
        anchor_mode = str(getattr(self.args, "async_shadow_anchor_mode", "sample"))
        if anchor_mode == "map":
            next_tok = torch.argmax(q_dist).view(1, 1).to(torch.long)
        else:
            next_tok = sample(
                logits[:, -1, :], self.args.temperature,
                self.args.top_k, self.args.top_p, generator=self.shadow_rng,
            ).cpu()
        tok_id = int(next_tok.item())
        return {
            "token": next_tok.detach().cpu(),
            "token_id": tok_id,
            "prob": float(q_dist[tok_id].item()),
            "mode": anchor_mode,
        }

    @torch.no_grad()
    def prepare_shadow_candidate(self, x_draft: torch.Tensor, shadow_gamma: int, cancel_event=None) -> Dict:
        """
        基于当前轮 draft 的 accept-all 假设，构造下一轮的异步 shadow draft。
        仅当 BS 返回的 x_{gamma+1} 与本地锚点 token 一致时，shadow 才可复用。
        """
        shadow_start = time.time()
        if cancel_event is not None and cancel_event.is_set():
            return {"cancelled": True, "shadow_compute_time_s": 0.0}
        anchor = self.sample_next_token_info(x_draft)
        if cancel_event is not None and cancel_event.is_set():
            return {"cancelled": True, "shadow_compute_time_s": time.time() - shadow_start}
        optimistic_prefix = torch.cat([
            x_draft.detach().to(self.device),
            anchor["token"].view(1, 1).to(self.device),
        ], dim=1)
        draft_rng = torch.Generator(device=self.rng_device)
        draft_rng.set_state(self.main_rng.get_state())
        rng_state_history = []
        progress_counter = {"tokens": 0}
        draft_result = self.draft_DSSD(
            optimistic_prefix, shadow_gamma, generator=draft_rng,
            cancel_event=cancel_event, rng_state_history=rng_state_history,
            progress_counter=progress_counter,
        )
        if draft_result is None:
            return {
                "cancelled": True,
                "shadow_compute_time_s": time.time() - shadow_start,
                "shadow_tokens_computed": int(progress_counter["tokens"]),
            }
        x_shadow, q_values_shadow, q_probs_shadow, _ = draft_result
        return {
            "cancelled": False,
            "anchor_token_id": int(anchor["token_id"]),
            "optimistic_prefix": optimistic_prefix.detach(),
            "x_shadow": x_shadow.detach(),
            "q_values": list(q_values_shadow),
            "q_probs": q_probs_shadow.detach().cpu(),
            "shadow_gamma": int(shadow_gamma),
            "shadow_compute_time_s": time.time() - shadow_start,
            "shadow_tokens_computed": int(progress_counter["tokens"]),
            "main_rng_state_history": rng_state_history,
        }

    def materialize_shadow_for_gamma(self, shadow: Dict, gamma: int) -> Tuple[torch.Tensor, List[float], torch.Tensor, int]:
        """
        Materialize a cached shadow at the requested gamma.

        A shorter request truncates the cache. A longer request commits the
        cached RNG prefix and drafts only the missing suffix synchronously.
        """
        gamma = int(gamma)
        optimistic_prefix = shadow["optimistic_prefix"].to(self.device)
        x_shadow = shadow["x_shadow"].to(self.device)
        prefix_len = optimistic_prefix.shape[1]
        generated_gamma = int(shadow.get("shadow_gamma", len(shadow["q_values"])))
        rng_states = shadow.get("main_rng_state_history", [])

        reused_gamma = min(gamma, generated_gamma)
        if len(rng_states) < reused_gamma:
            raise RuntimeError("shadow RNG state history is shorter than the reused prefix")
        self.main_rng.set_state(rng_states[reused_gamma - 1])

        x_draft = x_shadow[:, :prefix_len + reused_gamma]
        q_values = list(shadow["q_values"][:reused_gamma])
        q_probs = shadow["q_probs"][:reused_gamma].to(torch.float32)
        if gamma > generated_gamma:
            extra_gamma = gamma - generated_gamma
            extra_x, extra_values, extra_probs, _ = self.draft_DSSD(
                x_draft, extra_gamma, generator=self.main_rng
            )
            x_draft = extra_x
            q_values.extend(extra_values)
            q_probs = torch.cat([q_probs, extra_probs.to(torch.float32)], dim=0)

        meta_bytes = 16
        dup_bytes = meta_bytes + gamma * 4 + gamma * 4
        return x_draft, q_values, q_probs, dup_bytes

    def resample_DSSD(self, j: int, reject_packet: Dict, q_probs: torch.Tensor) -> torch.Tensor:
        """
        DSSD Resample阶段：设备用本地保存的 Q_j 分布和基站发送的 reject packet 重新采样
        参数：
          j: 拒绝位置（1~gamma）
          reject_packet: 基站返回的压缩/完整 packet
          q_probs: 设备保存的Q分布（shape: (gamma, V)）
        返回：
          xj_prime: 重新采样的token（shape: (1, 1)）
        """
        q_j = q_probs[j-1].to(torch.float32).cpu()
        pj_hat = reconstruct_pj_from_packet(reject_packet)  # 可能是稀疏近似，也可能是完整分布
        diff = (pj_hat - q_j).clamp(min=0)
        diff_sum = diff.sum()

        # 压缩过强时，可能导致 top-k 内没有足够的正差分质量；做一个保底回退
        if diff_sum <= 1e-12:
            pj_sum = pj_hat.sum()
            if pj_sum <= 1e-12:
                # 极端情况：直接在 Q_j 的 top-1 之外回退为 argmax(P_j_hat)
                tok = int(torch.argmax(q_j).item())
                return torch.tensor([[tok]], dtype=torch.long)
            diff = pj_hat / pj_sum
        else:
            diff = diff / diff_sum

        xj_prime = torch.multinomial(diff, 1).unsqueeze(0)  # 采样 x_j'
        return xj_prime


class BSNode:
    """基站节点 - 负责verify阶段的大模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
    
    def verify_step(self, x_draft, q_steps, gamma):
        """
        BS端执行verify步骤，验证UAV生成的候选token
        返回：
          n: 接受的token位置
          t_corr: 校正token
          correct_num: 接受的token数量  
          reject_num: 拒绝的token数量
        """
        correct_num = reject_num = 0
        prefix_len = x_draft.size(1) - gamma
        
        # 1) 大模型一次前向
        p_all = self.model(x_draft.to(self.device)).logits.cpu()  # (1, prefix_len+γ, V)
        p_slice = p_all[0, prefix_len-1 : prefix_len+gamma-1, :]  # 取 γ 行 -> (γ, V)

        # 2) softmax归一化
        p_probs = sampling_probs(
            p_slice, self.args.temperature, self.args.top_k, self.args.top_p
        )
        q_probs = sampling_probs(
            q_steps, self.args.temperature, self.args.top_k, self.args.top_p
        )

        # 3) accept/reject判断
        for i in range(gamma):
            tok_id = int(x_draft[0, prefix_len+i].item())
            if torch.rand(1).item() > (p_probs[i, tok_id] / q_probs[i, tok_id]):
                # 首次拒绝 → 回滚到 prefix_len+i-1
                n = prefix_len + i - 1
                diff = (p_probs[i] - q_probs[i]).clamp(min=0)
                diff = diff / diff.sum()
                t_corr = torch.multinomial(diff, 1).unsqueeze(0)  # (1,1)
                reject_num += 1
                return n, t_corr, correct_num, reject_num
            else:
                correct_num += 1
        
        # 全部通过：n = 最后位置
        n = prefix_len + gamma - 1
        prob_last = sampling_probs(
            p_all[0, n].unsqueeze(0), self.args.temperature,
            self.args.top_k, self.args.top_p
        )[0]
        t_corr = torch.multinomial(prob_last, 1).unsqueeze(0)  # shape (1,1)
        return n, t_corr, correct_num, reject_num
    
    def verify_DSSD(self, x_draft: torch.Tensor, q_values: List[float], gamma: int) -> Tuple[int, int, Dict, torch.Tensor, int, int]:
        """
        DSSD Verification阶段：用设备发送的概率值验证推测token。
        优化点：
          - 拒绝时不再默认返回完整 P_j
          - 优先构造压缩 reject packet；质量不足时自动回退为 full packet
        返回：
          j: 处理到的位置（1~gamma+1）
          flag: 是否拒绝（0=拒绝，1=接受）
          reject_packet: 拒绝位置的压缩/完整 packet（仅 flag=0 有效）
          xj: 接受时的新采样 token（仅 flag=1 有效）
          correct_num: 接受的 token 数量
          reject_num: 拒绝的 token 数量
        """
        correct_num = 0
        reject_num = 0
        prefix_len = x_draft.size(1) - gamma

        with torch.no_grad():
            # Keep edge-side logits and softmax on the target device. Moving the
            # full vocabulary tensor to CPU makes CPU drafting contend with edge
            # verification and defeats the intended compute overlap.
            p_all = self.model(x_draft.to(self.device)).logits
        p_logits = p_all[0, prefix_len-1 : prefix_len+gamma, :]
        p_probs = sampling_probs(
            p_logits, self.args.temperature, self.args.top_k, self.args.top_p
        )

        flag = 1
        j = 1
        reject_packet = None
        xj = None

        for i in range(gamma):
            current_j = i + 1
            tok_id = int(x_draft[0, prefix_len + i].item())
            q_i = max(float(q_values[i]), 1e-12)  # 避免除 0
            p_i = float(p_probs[i, tok_id].item())

            if torch.rand(1).item() > min(1.0, p_i / q_i):
                flag = 0
                j = current_j
                reject_packet = build_reject_packet(p_probs[i].to(torch.float32), self.args)
                reject_num += 1
                break
            else:
                correct_num += 1

        if flag == 1:
            j = gamma + 1
            # Transfer only the one distribution needed for the returned token;
            # this preserves the existing CPU RNG stream without copying all logits.
            xj = torch.multinomial(
                p_probs[gamma].to(torch.float32).cpu(), 1
            ).unsqueeze(0)

        return j, flag, reject_packet, xj, correct_num, reject_num

class DynamicDSSDController:
    """
    Risk-aware online controller for Dynamic DSSD.
    动态联合控制：
      - draft length gamma
      - reject packet compression on/off
      - reject_topk
      - reject_mass_threshold
    """

    def __init__(self, args: argparse.Namespace):
        self.min_gamma = max(1, int(getattr(args, "dynamic_min_gamma", 1)))
        parsed_gamma_set = self._parse_int_list(getattr(args, "dynamic_gamma_set", "1,2,3,4,6,8"), default=[1, 2, 3, 4, 6, 8])
        self.gamma_set = [g for g in parsed_gamma_set if g >= self.min_gamma]
        if not self.gamma_set:
            self.gamma_set = [self.min_gamma]
        self.topk_set = self._parse_int_list(getattr(args, "dynamic_reject_topk_set", "16,32,64,128"), default=[16, 32, 64, 128])
        self.mass_set = self._parse_float_list(getattr(args, "dynamic_reject_mass_set", "0.98,0.99,0.995"), default=[0.98, 0.99, 0.995])
        self.dtype = getattr(args, "reject_packet_dtype", "float16")
        self.enable_joint_packet_control = getattr(args, "dynamic_joint_packet_control", True)
        self.beta = getattr(args, "dynamic_beta", 0.2)
        self.z_alpha = getattr(args, "dynamic_z_alpha", 0.7)
        alpha_weights = [
            float(getattr(args, "dynamic_alpha_weight_history", 0.65)),
            float(getattr(args, "dynamic_alpha_weight_difficulty", 0.20)),
            float(getattr(args, "dynamic_alpha_weight_early_reject", 0.15)),
        ]
        alpha_weight_sum = sum(alpha_weights)
        if alpha_weight_sum <= 0.0:
            raise ValueError("Acceptance-prediction weights must have positive sum")
        (
            self.alpha_weight_history,
            self.alpha_weight_difficulty,
            self.alpha_weight_early_reject,
        ) = [weight / alpha_weight_sum for weight in alpha_weights]
        compression_weights = [
            float(getattr(args, "dynamic_cmp_weight_mass", 0.45)),
            float(getattr(args, "dynamic_cmp_weight_saving", 0.25)),
            float(getattr(args, "dynamic_cmp_weight_topk", 0.20)),
            float(getattr(args, "dynamic_cmp_weight_easiness", 0.10)),
        ]
        compression_weight_sum = sum(compression_weights)
        if compression_weight_sum <= 0.0:
            raise ValueError("Compression-proxy weights must have positive sum")
        (
            self.cmp_weight_mass,
            self.cmp_weight_saving,
            self.cmp_weight_topk,
            self.cmp_weight_easiness,
        ) = [weight / compression_weight_sum for weight in compression_weights]
        self.cmp_threshold_penalty = float(
            getattr(args, "dynamic_cmp_threshold_penalty", 0.15)
        )
        self.cmp_k_ref = max(1.0, float(getattr(args, "dynamic_cmp_k_ref", 64.0)))
        self.cmp_tau0 = float(getattr(args, "dynamic_cmp_tau0", 0.98))
        self.cmp_delta_tau = max(
            1e-8, float(getattr(args, "dynamic_cmp_delta_tau", 0.02))
        )
        self.lambda_sw = getattr(args, "dynamic_lambda_sw", 0.02)
        self.lambda_cmp = getattr(args, "dynamic_lambda_cmp", 0.02)
        self.lambda_tail = getattr(args, "dynamic_lambda_tail", 0.04)
        self.epsilon = getattr(args, "dynamic_hysteresis", 0.01)
        self.min_residency = getattr(args, "dynamic_min_residency", 2)
        self.alpha_panic = getattr(args, "dynamic_alpha_panic", 0.35)
        self.gamma_safe = max(self.min_gamma, getattr(args, "dynamic_safe_gamma", self.min_gamma))
        self.downlink_budget_frac = getattr(args, "dynamic_downlink_budget_frac", 0.45)
        self.warmup_rounds = max(0, int(getattr(args, "dynamic_warmup_rounds", 4)))
        self.panic_reject_rate = float(getattr(args, "dynamic_panic_reject_rate", 0.55))
        self.panic_early_reject = float(getattr(args, "dynamic_panic_early_reject", 0.45))
        self.debug = bool(getattr(args, "dynamic_debug", False))
        self.score_mode = getattr(args, "dynamic_score_mode", "ratio")
        self.compression_estimator = getattr(args, "dynamic_compression_estimator", "legacy")
        self.acceptance_predictor = getattr(args, "dynamic_acceptance_predictor", "legacy")
        if self.score_mode not in ("ratio", "difference"):
            raise ValueError("Unknown controller score mode")
        if self.compression_estimator not in ("legacy", "history"):
            raise ValueError("Unknown compression estimator")
        if self.acceptance_predictor not in ("legacy", "lcb"):
            raise ValueError("Unknown acceptance predictor")
        self.throughput_estimate = 0.0
        self.score_reference_time = 1.0
        self.compression_history = {}
        self.compression_observation_count = 0
        self.compression_brier_sum = 0.0

        self.alpha_ema = 0.6
        self.alpha_var = 0.0
        self.alpha_update_count = 0
        self.reject_rate_ema = 0.0
        self.early_reject_ema = 0.0
        self.difficulty_ema = 0.5
        self.kept_mass_ema = 1.0
        self.tail_mass_ema = 0.0
        self.saving_ema = 0.0
        self.t_slm_per_tok = 0.0
        self.t_llm_per_gamma = {}
        self.prev_action = {
            "gamma": min(self.gamma_set, key=lambda x: abs(x - getattr(args, "gamma", 4))),
            "compress_reject_packet": getattr(args, "compress_reject_packet", True),
            "reject_topk": getattr(args, "reject_topk", 64),
            "reject_mass_threshold": getattr(args, "reject_mass_threshold", 0.995),
            "reject_packet_dtype": self.dtype,
        }
        self.hold_rounds = 0
        self.reject_streak = 0
        self.round_idx = 0
        self.switch_count = 0
        self.safe_mode_count = 0
        self.warmup_count = 0
        self.hold_count = 0
        self.last_round_alpha = self.alpha_ema

    @staticmethod
    def _parse_int_list(s: str, default):
        try:
            vals = [int(x.strip()) for x in str(s).split(',') if x.strip()]
            return sorted(set(vals)) if vals else default
        except Exception:
            return default

    @staticmethod
    def _parse_float_list(s: str, default):
        try:
            vals = [float(x.strip()) for x in str(s).split(',') if x.strip()]
            return sorted(set(vals)) if vals else default
        except Exception:
            return default

    def alpha_std(self):
        return math.sqrt(max(self.alpha_var, 0.0))

    def alpha_lcb(self):
        if self.alpha_update_count <= 0:
            return max(0.05, min(0.999, self.alpha_ema))
        steady_state_n = (2.0 - self.beta) / max(self.beta, 1e-8)
        effective_n = max(1.0, min(float(self.alpha_update_count), steady_state_n))
        standard_error = self.alpha_std() / math.sqrt(effective_n)
        return max(0.05, min(0.999, self.alpha_ema - self.z_alpha * standard_error))

    def predict_alpha(self, difficulty: float, jitter: float):
        if self.acceptance_predictor == "lcb":
            return self.alpha_lcb()
        a = (
            self.alpha_weight_history * self.alpha_lcb()
            + self.alpha_weight_difficulty * (1.0 - difficulty)
            + self.alpha_weight_early_reject * (1.0 - self.early_reject_ema)
        )
        a -= 0.12 * jitter
        return max(0.05, min(0.999, a))

    def expected_advance(self, alpha: float, gamma: int) -> float:
        if abs(alpha - 1.0) < 1e-8:
            return gamma + 1.0
        return (1.0 - alpha ** (gamma + 1)) / max(1e-8, 1.0 - alpha)

    def estimate_t_llm(self, gamma: int) -> float:
        if gamma in self.t_llm_per_gamma:
            return self.t_llm_per_gamma[gamma]
        if not self.t_llm_per_gamma:
            return 0.0

        keys = sorted(self.t_llm_per_gamma.keys())
        if len(keys) == 1:
            return self.t_llm_per_gamma[keys[0]]

        if gamma <= keys[0]:
            return self.t_llm_per_gamma[keys[0]]
        if gamma >= keys[-1]:
            k0, k1 = keys[-2], keys[-1]
        else:
            k0, k1 = keys[0], keys[-1]
            for i in range(len(keys) - 1):
                if keys[i] <= gamma <= keys[i + 1]:
                    k0, k1 = keys[i], keys[i + 1]
                    break
        if k0 == k1:
            return self.t_llm_per_gamma[k0]
        v0, v1 = self.t_llm_per_gamma[k0], self.t_llm_per_gamma[k1]
        return v0 + (v1 - v0) * (gamma - k0) / (k1 - k0)

    def up_bytes(self, gamma: int) -> int:
        return 16 + gamma * 4 + gamma * 4

    def down_bytes_accept(self) -> int:
        return 8

    def down_bytes_reject(self, compress: bool, topk: int, mass: float, vocab_size: int) -> int:
        # 压缩成功时按 ids(int32)+probs(dtype)+meta；full packet 使用同一概率精度。
        prob_b = 2 if self.dtype in ("float16", "bfloat16") else 4
        full = 16 + vocab_size * prob_b
        if not compress:
            return full
        # 质量阈值越高，压缩命中率越低；difficulty/recent stats 由外层概率建模吸收。
        # float16/bfloat16 按 2 bytes, float32 按 4 bytes
        comp = 16 + topk * 4 + topk * prob_b
        return min(comp, full)

    def predict_compression_success(self, difficulty: float, topk: int, mass: float) -> float:
        if self.compression_estimator == "history":
            # Only actual rejects at this configuration update its probability.
            return self.compression_history.get((int(topk), float(mass)), 0.5)
        topk_factor = min(1.0, topk / self.cmp_k_ref)
        threshold_strictness = max(
            0.0, min(1.0, (mass - self.cmp_tau0) / self.cmp_delta_tau)
        )
        p = (
            self.cmp_weight_mass * self.kept_mass_ema
            + self.cmp_weight_saving * self.saving_ema
            + self.cmp_weight_topk * topk_factor
            + self.cmp_weight_easiness * (1.0 - difficulty)
            - self.cmp_threshold_penalty * threshold_strictness
        )
        return max(0.0, min(1.0, p))

    def predicted_terms(self, action: Dict, state: Dict):
        gamma = action["gamma"]
        difficulty = state["difficulty"]
        jitter = state["jitter"]
        vocab_size = state["vocab_size"]
        alpha = self.predict_alpha(difficulty, jitter)
        adv = self.expected_advance(alpha, gamma)

        up_delay_fn = state.get("up_delay_fn", state["tx_delay_fn"])
        down_delay_fn = state.get("down_delay_fn", state["tx_delay_fn"])
        up_t = up_delay_fn(self.up_bytes(gamma))
        if action["compress_reject_packet"]:
            p_cmp = self.predict_compression_success(difficulty, action["reject_topk"], action["reject_mass_threshold"])
        else:
            p_cmp = 0.0
        d_rej_cmp = self.down_bytes_reject(True, action["reject_topk"], action["reject_mass_threshold"], vocab_size)
        d_rej_full = self.down_bytes_reject(False, action["reject_topk"], action["reject_mass_threshold"], vocab_size)
        d_acc = self.down_bytes_accept()
        down_bytes = (alpha ** gamma) * d_acc + (1.0 - alpha ** gamma) * (p_cmp * d_rej_cmp + (1.0 - p_cmp) * d_rej_full)
        down_t = down_delay_fn(down_bytes)
        t_round = gamma * self.t_slm_per_tok + self.estimate_t_llm(gamma) + up_t + down_t
        return alpha, adv, p_cmp, t_round

    def score(self, action: Dict, state: Dict) -> float:
        gamma = action["gamma"]
        alpha, adv, p_cmp, t_round = self.predicted_terms(action, state)
        p_early = 1.0 - alpha ** min(gamma, 2)
        p_rej = 1.0 - alpha ** gamma
        switch_penalty = self.lambda_sw * abs(gamma - self.prev_action["gamma"])
        cmp_risk = 0.0
        if action["compress_reject_packet"]:
            cmp_risk = self.lambda_cmp * ((1.0 - self.kept_mass_ema) + self.tail_mass_ema + (1.0 - p_cmp))
        tail_risk = self.lambda_tail * (0.7 * p_early + 0.3 * p_rej)
        if self.score_mode == "difference":
            # A shared time scale preserves token/s penalty units for all actions.
            utility = (adv - self.throughput_estimate * t_round) / self.score_reference_time
        else:
            utility = adv / max(t_round, 1e-8)
        return utility - switch_penalty - cmp_risk - tail_risk

    def hysteresis_margin(self, prev_score: float) -> float:
        if self.score_mode == "difference":
            # Difference scores can cross zero; use a positive throughput scale.
            return self.epsilon * max(self.throughput_estimate, 1.0 / self.score_reference_time)
        return self.epsilon * max(abs(prev_score), 1e-8)

    def choose_action(self, state: Dict) -> Dict:
        # Collect acceptance and latency observations before optimizing the
        # one-step utility. With an empty cost model, the utility otherwise
        # favors the largest gamma because local/edge compute appears free.
        if self.round_idx < self.warmup_rounds or (
            self.score_mode == "difference" and self.throughput_estimate <= 0.0
        ):
            warmup_action = dict(self.prev_action)
            warmup_action["gamma"] = self.gamma_safe
            self.prev_action = warmup_action
            self.warmup_count += 1
            return dict(warmup_action)

        alpha_lcb = self.alpha_lcb()
        panic_due_to_low_confidence = (
            alpha_lcb < self.alpha_panic
            and self.reject_rate_ema > self.panic_reject_rate
            and self.early_reject_ema > self.panic_early_reject
        )
        panic_trigger = self.round_idx >= self.warmup_rounds and panic_due_to_low_confidence
        if panic_trigger:
            old_action = dict(self.prev_action)
            safe = dict(old_action)
            # Safe mode limits speculative compute. Reject-packet quality is
            # already protected by the retained-mass gate, so disabling sparse
            # feedback here would increase downlink cost exactly when rejects
            # are frequent.
            safe["gamma"] = self.gamma_safe
            if safe != old_action:
                self.switch_count += 1
            self.safe_mode_count += 1
            self.prev_action = safe
            self.hold_rounds = self.min_residency
            if self.debug:
                reasons = []
                if panic_due_to_low_confidence:
                    reasons.append(
                        f"alpha_lcb={alpha_lcb:.3f}<panic={self.alpha_panic:.3f}, reject_rate_ema={self.reject_rate_ema:.3f}, early_reject_ema={self.early_reject_ema:.3f}"
                    )
                print(
                    f"[DynamicCtrl] SAFE_MODE round={self.round_idx} gamma->{safe['gamma']} "
                    f"compress={int(bool(safe['compress_reject_packet']))} "
                    f"difficulty={state['difficulty']:.3f} prev_gamma={old_action['gamma']} reason={' | '.join(reasons)}"
                )
            return safe

        candidates = []
        for gamma in self.gamma_set:
            if self.enable_joint_packet_control:
                candidates.append({
                    "gamma": gamma,
                    "compress_reject_packet": False,
                    "reject_topk": self.prev_action["reject_topk"],
                    "reject_mass_threshold": self.prev_action["reject_mass_threshold"],
                    "reject_packet_dtype": self.dtype,
                })
                for topk in self.topk_set:
                    for mass in self.mass_set:
                        candidates.append({
                            "gamma": gamma,
                            "compress_reject_packet": True,
                            "reject_topk": topk,
                            "reject_mass_threshold": mass,
                            "reject_packet_dtype": self.dtype,
                        })
            else:
                candidates.append({
                    "gamma": gamma,
                    "compress_reject_packet": self.prev_action["compress_reject_packet"],
                    "reject_topk": self.prev_action["reject_topk"],
                    "reject_mass_threshold": self.prev_action["reject_mass_threshold"],
                    "reject_packet_dtype": self.dtype,
                })

        if self.score_mode == "difference":
            self.score_reference_time = max(self.predicted_terms(self.prev_action, state)[3], 1e-8)
        scored = [(self.score(a, state), a) for a in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_action = scored[0]
        prev_score = self.score(self.prev_action, state)

        if self.hold_rounds > 0:
            self.hold_rounds -= 1
            self.hold_count += 1
            if self.debug:
                print(
                    f"[DynamicCtrl] HOLD round={self.round_idx} gamma={self.prev_action['gamma']} compress={int(bool(self.prev_action['compress_reject_packet']))} "
                    f"hold_left={self.hold_rounds} alpha_lcb={alpha_lcb:.3f} reject_rate_ema={self.reject_rate_ema:.3f} early_reject_ema={self.early_reject_ema:.3f}"
                )
            return dict(self.prev_action)

        # Interpret hysteresis as a relative utility improvement. An absolute
        # threshold is hardware-dependent because Score is measured in tok/s.
        hysteresis_margin = self.hysteresis_margin(prev_score)
        if best_score > prev_score + hysteresis_margin:
            old_action = dict(self.prev_action)
            self.prev_action = dict(best_action)
            self.hold_rounds = self.min_residency
            self.switch_count += 1
            if self.debug:
                print(
                    f"[DynamicCtrl] SWITCH round={self.round_idx} gamma {old_action['gamma']}->{best_action['gamma']} "
                    f"compress {int(bool(old_action['compress_reject_packet']))}->{int(bool(best_action['compress_reject_packet']))} "
                    f"score {prev_score:.4f}->{best_score:.4f} alpha_lcb={alpha_lcb:.3f} difficulty={state['difficulty']:.3f}"
                )
        elif self.debug:
            print(
                f"[DynamicCtrl] KEEP round={self.round_idx} gamma={self.prev_action['gamma']} compress={int(bool(self.prev_action['compress_reject_packet']))} "
                f"prev_score={prev_score:.4f} best_score={best_score:.4f} alpha_lcb={alpha_lcb:.3f} difficulty={state['difficulty']:.3f}"
            )
        return dict(self.prev_action)

    def update(self, obs: Dict):
        if self.score_mode == "difference":
            elapsed = float(obs.get("elapsed_s", 0.0))
            generated = int(obs.get("generated_tokens", 0))
            if not math.isfinite(elapsed) or elapsed <= 0.0 or generated <= 0:
                raise ValueError("Difference scoring requires positive cumulative time and committed tokens")
            self.throughput_estimate = generated / elapsed
        if self.compression_estimator == "history" and (
            int(obs.get("flag", 1)) == 0 and bool(obs.get("compression_attempted", False))
        ):
            key = (int(obs["reject_topk"]), float(obs["reject_mass_threshold"]))
            prediction = self.compression_history.get(key, 0.5)
            outcome = float(bool(obs["compressed_feedback"]))
            self.compression_brier_sum += (prediction - outcome) ** 2
            self.compression_observation_count += 1
            self.compression_history[key] = (1.0 - self.beta) * prediction + self.beta * outcome
        self.round_idx += 1
        gamma = max(int(obs["gamma"]), 1)
        accepted = max(int(obs["correct_num"]), 0)
        rejected = 1 if int(obs.get("flag", 0)) == 0 else 0
        verified = max(accepted + rejected, 1)
        # The suffix after the first rejection is not verified. Counting that
        # suffix as rejected biases the controller's Bernoulli acceptance
        # estimate downward, especially for large gamma and early rejection.
        round_alpha = float(accepted) / verified
        prev_alpha = self.alpha_ema
        delta = round_alpha - prev_alpha
        self.alpha_ema = prev_alpha + self.beta * delta
        self.alpha_var = (1.0 - self.beta) * (
            self.alpha_var + self.beta * delta * delta
        )
        self.alpha_update_count += 1
        self.last_round_alpha = round_alpha
        reject_flag = 1.0 if obs["flag"] == 0 else 0.0
        self.reject_rate_ema = (1.0 - self.beta) * self.reject_rate_ema + self.beta * reject_flag

        early_reject = 1.0 if (obs["flag"] == 0 and int(obs["j"]) <= 2) else 0.0
        self.early_reject_ema = (1.0 - self.beta) * self.early_reject_ema + self.beta * early_reject
        self.reject_streak = self.reject_streak + 1 if reject_flag > 0.5 else 0

        difficulty = float(obs.get("difficulty", 0.5))
        self.difficulty_ema = (1.0 - self.beta) * self.difficulty_ema + self.beta * difficulty

        t_slm_per_tok = float(obs.get("t_slm", 0.0)) / gamma
        self.t_slm_per_tok = (1.0 - self.beta) * self.t_slm_per_tok + self.beta * t_slm_per_tok

        t_llm = float(obs.get("t_llm", 0.0))
        old_llm = self.t_llm_per_gamma.get(gamma, t_llm)
        self.t_llm_per_gamma[gamma] = (1.0 - self.beta) * old_llm + self.beta * t_llm

        if int(obs.get("flag", 1)) == 0 and bool(obs.get("compression_attempted", True)):
            kept_mass = float(obs.get("candidate_kept_mass", obs.get("kept_mass", 1.0)))
            tail_mass = float(obs.get("candidate_tail_mass", obs.get("tail_mass", 0.0)))
            saving = float(obs.get("saving", 0.0))
            self.kept_mass_ema = (1.0 - self.beta) * self.kept_mass_ema + self.beta * kept_mass
            self.tail_mass_ema = (1.0 - self.beta) * self.tail_mass_ema + self.beta * tail_mass
            self.saving_ema = (1.0 - self.beta) * self.saving_ema + self.beta * saving


def generate_DSD(uav_node, bs_node, input_ids, tokenizer, args):
    total_comm_raw = 0.0
    total_comm_comp = 0.0
    """
    分布式投机采样主循环，使用UAV和BS节点协作
    """
    input_ids = input_ids.to(uav_node.device)
    
    total_comm = total_slm = total_llm = 0.0
    rounds = correct_nums = reject_nums = 0
    total_dup_bytes = 0
    round_total_latency_hist = []
    round_comm_latency_hist = []
    round_slm_latency_hist = []
    round_llm_latency_hist = []
    gamma_hist = []
    total_proposed_tokens = 0
    
    # === 新增：分别统计两种上传方案的通信耗时 ===
    total_comm_raw  = 0.0   # 直接上传完整 logits
    total_comm_comp = 0.0   # 上传 dist-summary
    
    # 准备参数
    max_total_len = args.max_len + input_ids.shape[1]
    rtt = args.rtt
    torch.manual_seed(args.seed)
    prefix = input_ids

    # 主循环
    with tqdm(total=max_total_len, desc="distributed speculative sampling") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        dsp_start = time.time()
        
        while prefix.shape[1] < max_total_len:
            round_start = time.time()
            old_len = prefix.shape[1]
            remain = max_total_len - old_len
            gamma_t = min(int(args.gamma), remain)
            gamma_hist.append(gamma_t)
            total_proposed_tokens += gamma_t
            rounds += 1
            
            # UAV端：执行draft步骤
            t0 = time.time()
            x_draft, q_probs, dup_bytes = uav_node.draft_step(prefix, gamma_t)                  # 设备A
            total_dup_bytes += dup_bytes
            
            # 模拟上行传输延迟
            bw_Bps = args.bandwidth * 1e6 / 8  # Mbps → B/s
            t_up = tx_delay_bytes(dup_bytes, rtt, bw_Bps)
            
            # ---- 计算本轮两种方案的通信时延（不影响原有 sleep） ----
            raw_bytes  = tensor_nbytes(q_probs)
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_probs)
            comm_raw   = tx_delay_bytes(raw_bytes,  rtt, bw_Bps)
            comm_comp  = tx_delay_bytes(comp_bytes, rtt, bw_Bps)
            total_comm_raw  += comm_raw
            total_comm_comp += comm_comp
            
            round_slm = time.time() - t0
            total_slm += round_slm
            total_comm += t_up
            time.sleep(t_up)

            # BS端：执行verify步骤
            t1 = time.time()
            n, t_corr, correct_num, reject_num = bs_node.verify_step(                           # 设备B
                x_draft, q_probs, gamma_t)
            correct_nums += correct_num
            reject_nums += reject_num
            round_llm = time.time() - t1
            total_llm += round_llm
            
            # === 新增：统计下行通信时间（t_corr从BS返回到UAV）===
            down_bytes = tensor_nbytes(t_corr)          # 1 token → 4 bytes
            t_down = tx_delay_bytes(down_bytes, rtt, bw_Bps)
            total_comm += t_down
            total_comm_raw  += t_down                   # 两种方案下行相同
            total_comm_comp += t_down
            time.sleep(t_down)
            
            # UAV端：接收结果并更新prefix
            prefix = torch.cat([
                x_draft[:, : n+1],
                t_corr.to(uav_node.device)
            ], dim=1)[:, :max_total_len]

            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            round_total_latency_hist.append(time.time() - round_start)
            round_comm_latency_hist.append(t_up + t_down)
            round_slm_latency_hist.append(round_slm)
            round_llm_latency_hist.append(round_llm)
            
    dsp_time = time.time() - dsp_start
    total_tokens = prefix.shape[1] - initial_len
    dsp_throughput = total_tokens / dsp_time
    acceptance_rate = correct_nums / max(total_proposed_tokens, 1)

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== Distributed SP Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{dsp_throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total accepted tokens: {correct_nums}")
    print(f"Total rejected tokens: {reject_nums}")
    print(f"Total proposed tokens: {total_proposed_tokens}")
    print(f"Accept/Reject ratio: {correct_nums}/{reject_nums} = {correct_nums/max(reject_nums,1):.2f}")
    print(f"Communication time (raw logits)   : {total_comm_raw:.3f} s")
    print(f"Communication time (dist summary) : {total_comm_comp:.3f} s") 
    print(f"Time saved with dist summary      : {(total_comm_raw-total_comm_comp):.3f} s")
    
    result = (
        generated, dsp_throughput, dsp_time, acceptance_rate, total_comm,
        total_dup_bytes, rounds, total_slm, total_llm, total_comm_raw,
        total_comm_comp,
    )
    if not getattr(args, "return_runtime_stats", False):
        return result

    runtime_stats = {
        "accept_all_rounds": rounds - reject_nums,
        "reject_rounds": reject_nums,
        "avg_gamma": sum(gamma_hist) / max(len(gamma_hist), 1),
        "gamma_usage": _hist_counts(gamma_hist),
        "controller_switch_count": 0,
        "controller_switch_rate": 0.0,
    }
    runtime_stats.update(_latency_stats(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_stats(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_stats(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_stats(round_llm_latency_hist, "round_llm_latency"))
    runtime_stats.update(_latency_samples(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_samples(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_samples(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_samples(round_llm_latency_hist, "round_llm_latency"))
    return result + (runtime_stats,)

def generate_DSSD(uav_node: UAVNode, bs_node: BSNode, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace, run_tag: str = "DSSD", show_compression_stats: bool = True) -> Tuple[str, float, float, float, float, int, int, float, float, Dict]:
    """
    DSSD主循环：设备与基站协作生成文本（符合Algorithm 2）
    返回：
      generated: 生成的文本
      throughput: 吞吐量（tokens/s）
      total_time: 总时间（秒）
      acceptance_rate: 接受率
      total_comm_delay: 总通信延迟（秒）
      total_dup_bytes: 上行传输总字节数（字节）
      rounds: 循环轮次
      total_slm_time: 设备小模型总时间（秒）
      total_llm_time: 基站大模型总时间（秒）
    """
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]  # 生成的总长度（输入+输出）
    link_replay = TimeVaryingLinkReplay.from_args(args)

    # 初始化统计指标
    total_comm_delay = 0.0  # 总通信延迟（上行+下行）
    total_slm_time = 0.0    # 设备小模型（SLM）总时间
    total_llm_time = 0.0    # 基站大模型（LLM）总时间
    rounds = 0              # 循环轮次
    correct_num_total = 0   # 接受的token总数
    reject_num_total = 0    # 拒绝的token总数
    total_dup_bytes = 0     # 上行传输总字节数

    # ===== 新增：reject packet 统计 =====
    reject_compressed_cnt = 0
    reject_full_cnt = 0
    accept_all_rounds = 0
    reject_down_bytes_actual = 0
    reject_down_bytes_full_equiv = 0
    reject_kept_mass_sum = 0.0
    reject_tail_mass_sum = 0.0
    reject_candidate_kept_mass_sum = 0.0
    reject_candidate_tail_mass_sum = 0.0
    reject_candidate_mass_count = 0
    round_total_latency_hist = []
    round_comm_latency_hist = []
    round_slm_latency_hist = []
    round_llm_latency_hist = []
    gamma_hist = []
    total_proposed_tokens = 0

    # 初始化前缀
    prefix = input_ids
    torch.manual_seed(args.seed)

    with tqdm(total=max_total_len, desc=f"{run_tag}: Distributed Split Speculative Decoding") as pbar:
        pbar.update(prefix.shape[1])  # 初始进度（输入长度）
        initial_len = prefix.shape[1]
        start_time = time.time()

        # 主循环：直到生成达到最大长度
        while prefix.shape[1] < max_total_len:
            round_start = time.time()
            old_len = prefix.shape[1]
            remain = max_total_len - old_len
            gamma_t = min(int(args.gamma), remain)
            gamma_hist.append(gamma_t)
            total_proposed_tokens += gamma_t
            rounds += 1

            # 1. 设备端：Draft阶段（生成推测token+概率值）
            t_slm_start = time.time()
            # 调用修改后的draft_DSSD，获取token、概率值、完整分布、传输字节数
            x_draft, q_values, q_probs, dup_bytes = uav_node.draft_DSSD(prefix, gamma_t)
            round_slm = time.time() - t_slm_start
            total_dup_bytes += dup_bytes  # 统计上行传输总字节数（token+概率值）
            total_slm_time += round_slm

            # 模拟上行传输延迟（RTT/2 + 序列化延迟）
            t_up = link_replay.delay_bytes(dup_bytes, direction="up")
            total_comm_delay += t_up
            time.sleep(t_up)  # 模拟传输时间

            # 2. 基站端：Verification阶段（用概率值验证）
            t_llm_start = time.time()
            # 调用修改后的verify_DSSD，传递概率值q_values
            j, flag, reject_packet, xj, correct_num, reject_num = bs_node.verify_DSSD(x_draft, q_values, gamma_t)
            round_llm = time.time() - t_llm_start
            correct_num_total += correct_num
            reject_num_total += reject_num
            total_llm_time += round_llm

            # 3. 设备端：处理基站反馈（Resample + Reset）
            prefix_len = prefix.shape[1]  # 当前前缀长度
            if flag == 1:
                # 情况1：全部接受（flag=1）→ 基站返回x_{gamma+1}
                accept_all_rounds += 1
                new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
                down_bytes = 4 + xj.numel() * 4  # j占4字节，xj占4字节
            else:
                # 情况2：拒绝（flag=0）→ 基站返回压缩/完整 reject packet
                if reject_packet["mode"] == "compressed":
                    reject_compressed_cnt += 1
                else:
                    reject_full_cnt += 1
                reject_kept_mass_sum += float(reject_packet.get("kept_mass", 1.0))
                reject_tail_mass_sum += float(reject_packet.get("tail_mass", 0.0))
                if bool(getattr(args, "compress_reject_packet", False)):
                    reject_candidate_kept_mass_sum += float(
                        reject_packet.get("candidate_kept_mass", reject_packet.get("kept_mass", 1.0))
                    )
                    reject_candidate_tail_mass_sum += float(
                        reject_packet.get("candidate_tail_mass", reject_packet.get("tail_mass", 0.0))
                    )
                    reject_candidate_mass_count += 1

                packet_bytes_actual = reject_packet_nbytes(reject_packet)
                packet_bytes_full = full_packet_equivalent_nbytes(reject_packet)
                reject_down_bytes_actual += packet_bytes_actual
                reject_down_bytes_full_equiv += packet_bytes_full

                xj_prime = uav_node.resample_DSSD(j, reject_packet, q_probs)
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
                down_bytes = 4 + packet_bytes_actual

            new_prefix = new_prefix[:, :max_total_len]

            # 模拟下行传输延迟
            t_down = link_replay.delay_bytes(down_bytes, direction="down")
            total_comm_delay += t_down
            time.sleep(t_down)  # 模拟传输时间

            # 更新前缀
            prefix = new_prefix
            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            round_total_latency_hist.append(time.time() - round_start)
            round_comm_latency_hist.append(t_up + t_down)
            round_slm_latency_hist.append(round_slm)
            round_llm_latency_hist.append(round_llm)

    # 4. 结果统计
    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len  # 生成的token总数（排除输入）
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    acceptance_rate = correct_num_total / max(total_proposed_tokens, 1)

    # 解码生成的文本
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print(f"\n=== {run_tag} Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total accepted tokens: {correct_num_total}")
    print(f"Total rejected tokens: {reject_num_total}")
    print(f"Total proposed tokens: {total_proposed_tokens}")
    print(f"Accept/Reject ratio: {correct_num_total}/{reject_num_total} = {correct_num_total/max(reject_num_total,1):.2f}")
    print(f"Total communication delay: {total_comm_delay:.2f}s")
    print(f"Total SLM (device) time: {total_slm_time:.2f}s")
    print(f"Total LLM (BS) time: {total_llm_time:.2f}s")

    reject_rounds = max(reject_compressed_cnt + reject_full_cnt, 0)
    compression_hit_rate = (reject_compressed_cnt / reject_rounds) if reject_rounds > 0 else 0.0
    reject_downlink_saving = (
        1.0 - reject_down_bytes_actual / max(reject_down_bytes_full_equiv, 1)
    ) if reject_rounds > 0 else 0.0
    avg_kept_mass = reject_kept_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_tail_mass = reject_tail_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_candidate_kept_mass = reject_candidate_kept_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0
    avg_candidate_tail_mass = reject_candidate_tail_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0

    if show_compression_stats:
        print("\n--- Reject Packet Compression Stats ---")
        print(f"Accept-all rounds: {accept_all_rounds}")
        print(f"Reject rounds: {reject_rounds}")
        print(f"Compressed reject count: {reject_compressed_cnt}")
        print(f"Full reject count: {reject_full_cnt}")
        print(f"Compression hit rate: {compression_hit_rate:.3f}")
        print(f"Reject downlink bytes (actual): {reject_down_bytes_actual} B")
        print(f"Reject downlink bytes (full-equiv): {reject_down_bytes_full_equiv} B")
        print(f"Reject downlink saving: {reject_downlink_saving:.2%}")
        print(f"Average kept mass in reject packet: {avg_kept_mass:.6f}")
        print(f"Average tail mass in reject packet: {avg_tail_mass:.6f}")

    reject_stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "reject_compressed_cnt": reject_compressed_cnt,
        "reject_full_cnt": reject_full_cnt,
        "compression_hit_rate": compression_hit_rate,
        "reject_down_bytes_actual": reject_down_bytes_actual,
        "reject_down_bytes_full_equiv": reject_down_bytes_full_equiv,
        "reject_downlink_saving": reject_downlink_saving,
        "avg_kept_mass": avg_kept_mass,
        "avg_tail_mass": avg_tail_mass,
        "avg_candidate_kept_mass": avg_candidate_kept_mass,
        "avg_candidate_tail_mass": avg_candidate_tail_mass,
        "candidate_mass_samples": reject_candidate_mass_count,
        "avg_gamma": sum(gamma_hist) / max(len(gamma_hist), 1),
        "gamma_usage": _hist_counts(gamma_hist),
    }
    reject_stats.update(_latency_stats(round_total_latency_hist, "round_total_latency"))
    reject_stats.update(_latency_stats(round_comm_latency_hist, "round_comm_latency"))
    reject_stats.update(_latency_stats(round_slm_latency_hist, "round_slm_latency"))
    reject_stats.update(_latency_stats(round_llm_latency_hist, "round_llm_latency"))
    reject_stats.update(_latency_samples(round_total_latency_hist, "round_total_latency"))
    reject_stats.update(_latency_samples(round_comm_latency_hist, "round_comm_latency"))
    reject_stats.update(_latency_samples(round_slm_latency_hist, "round_slm_latency"))
    reject_stats.update(_latency_samples(round_llm_latency_hist, "round_llm_latency"))

    return generated, throughput, total_time, acceptance_rate, total_comm_delay, total_dup_bytes, rounds, total_slm_time, total_llm_time, reject_stats
    

def generate_dynamic_DSSD(uav_node: UAVNode, bs_node: BSNode, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace, run_tag: str = "Dynamic DSSD"):
    """
    Dynamic DSSD: on top of Optimized DSSD, jointly adapt gamma and reject-packet compression policy online.
    """
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]
    link_replay = TimeVaryingLinkReplay.from_args(args)
    controller_link = ControllerLinkTelemetry(link_replay, args)

    controller = DynamicDSSDController(args)

    total_comm_delay = 0.0
    total_slm_time = 0.0
    total_llm_time = 0.0
    rounds = 0
    correct_num_total = 0
    reject_num_total = 0
    total_dup_bytes = 0
    total_proposed_tokens = 0

    reject_compressed_cnt = 0
    reject_full_cnt = 0
    accept_all_rounds = 0
    reject_down_bytes_actual = 0
    reject_down_bytes_full_equiv = 0
    reject_kept_mass_sum = 0.0
    reject_tail_mass_sum = 0.0
    reject_candidate_kept_mass_sum = 0.0
    reject_candidate_tail_mass_sum = 0.0
    reject_candidate_mass_count = 0

    dynamic_gamma_hist = []
    dynamic_topk_hist = []
    dynamic_mass_hist = []
    dynamic_compress_hist = []
    dynamic_reject_pos_hist = []
    difficulty_hist = []
    round_total_latency_hist = []
    round_comm_latency_hist = []
    round_slm_latency_hist = []
    round_llm_latency_hist = []
    preview_time_total = 0.0
    preview_fresh_count = 0
    controller_decision_time_total = 0.0

    prefix = input_ids
    torch.manual_seed(args.seed)

    with tqdm(total=max_total_len, desc=f"{run_tag}: Dynamic Split Speculative Decoding") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        start_time = time.time()

        while prefix.shape[1] < max_total_len:
            round_start = time.time()
            old_len = prefix.shape[1]
            preview_start = time.time()
            difficulty_info = uav_node.preview_difficulty(prefix)
            preview_time = time.time() - preview_start
            preview_time_total += preview_time
            preview_fresh_count += 1
            difficulty = float(difficulty_info["difficulty"])
            difficulty_hist.append(difficulty)

            state = {
                "difficulty": difficulty,
                "jitter": 0.0,
                "vocab_size": int(getattr(tokenizer, 'vocab_size', 32000)),
                "tx_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="down"),
                "up_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="up"),
                "down_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="down"),
            }
            decision_start = time.perf_counter()
            action = controller.choose_action(state)
            controller_decision_time_total += time.perf_counter() - decision_start

            gamma_t = int(action["gamma"])
            remain = max_total_len - prefix.shape[1]
            gamma_t = min(gamma_t, max(1, remain))

            uav_node.args.gamma = gamma_t
            bs_node.args.gamma = gamma_t
            uav_node.args.compress_reject_packet = bool(action["compress_reject_packet"])
            bs_node.args.compress_reject_packet = bool(action["compress_reject_packet"])
            uav_node.args.reject_topk = int(action["reject_topk"])
            bs_node.args.reject_topk = int(action["reject_topk"])
            uav_node.args.reject_mass_threshold = float(action["reject_mass_threshold"])
            bs_node.args.reject_mass_threshold = float(action["reject_mass_threshold"])
            uav_node.args.reject_packet_dtype = action["reject_packet_dtype"]
            bs_node.args.reject_packet_dtype = action["reject_packet_dtype"]

            dynamic_gamma_hist.append(gamma_t)
            dynamic_topk_hist.append(int(action["reject_topk"]))
            dynamic_mass_hist.append(float(action["reject_mass_threshold"]))
            dynamic_compress_hist.append(int(bool(action["compress_reject_packet"])))

            rounds += 1
            total_proposed_tokens += gamma_t

            t_slm_start = time.time()
            x_draft, q_values, q_probs, dup_bytes = uav_node.draft_DSSD(
                prefix, gamma_t, first_logits=difficulty_info["first_logits"]
            )
            round_slm = preview_time + (time.time() - t_slm_start)
            total_dup_bytes += dup_bytes
            total_slm_time += round_slm

            t_up = link_replay.delay_bytes(dup_bytes, direction="up")
            total_comm_delay += t_up
            time.sleep(t_up)

            t_llm_start = time.time()
            j, flag, reject_packet, xj, correct_num, reject_num = bs_node.verify_DSSD(x_draft, q_values, gamma_t)
            round_llm = time.time() - t_llm_start
            total_llm_time += round_llm
            correct_num_total += correct_num
            reject_num_total += reject_num

            prefix_len = prefix.shape[1]
            kept_mass = 1.0
            tail_mass = 0.0
            candidate_kept_mass = 1.0
            candidate_tail_mass = 0.0
            saving = 0.0

            if flag == 1:
                accept_all_rounds += 1
                new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
                down_bytes = 4 + xj.numel() * 4
                dynamic_reject_pos_hist.append(0)
            else:
                dynamic_reject_pos_hist.append(int(j))
                if reject_packet["mode"] == "compressed":
                    reject_compressed_cnt += 1
                else:
                    reject_full_cnt += 1
                kept_mass = float(reject_packet.get("kept_mass", 1.0))
                tail_mass = float(reject_packet.get("tail_mass", 0.0))
                candidate_kept_mass = float(reject_packet.get("candidate_kept_mass", kept_mass))
                candidate_tail_mass = float(reject_packet.get("candidate_tail_mass", tail_mass))
                reject_kept_mass_sum += kept_mass
                reject_tail_mass_sum += tail_mass
                if bool(action["compress_reject_packet"]):
                    reject_candidate_kept_mass_sum += candidate_kept_mass
                    reject_candidate_tail_mass_sum += candidate_tail_mass
                    reject_candidate_mass_count += 1

                packet_bytes_actual = reject_packet_nbytes(reject_packet)
                packet_bytes_full = full_packet_equivalent_nbytes(reject_packet)
                reject_down_bytes_actual += packet_bytes_actual
                reject_down_bytes_full_equiv += packet_bytes_full
                saving = 1.0 - packet_bytes_actual / max(packet_bytes_full, 1)

                xj_prime = uav_node.resample_DSSD(j, reject_packet, q_probs)
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
                down_bytes = 4 + packet_bytes_actual

            t_down = link_replay.delay_bytes(down_bytes, direction="down")
            total_comm_delay += t_down
            time.sleep(t_down)
            controller_link.observe_completed_round()

            prefix = new_prefix
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            round_total_latency_hist.append(time.time() - round_start)
            round_comm_latency_hist.append(t_up + t_down)
            round_slm_latency_hist.append(round_slm)
            round_llm_latency_hist.append(round_llm)

            controller.update({
                "elapsed_s": time.time() - start_time,
                "generated_tokens": new_len - initial_len,
                "reject_topk": action["reject_topk"],
                "reject_mass_threshold": action["reject_mass_threshold"],
                "compressed_feedback": flag == 0 and reject_packet["mode"] == "compressed",
                "gamma": gamma_t,
                "correct_num": correct_num,
                "flag": flag,
                "j": int(j),
                "difficulty": difficulty,
                "t_slm": round_slm,
                "t_llm": round_llm,
                "kept_mass": kept_mass,
                "tail_mass": tail_mass,
                "candidate_kept_mass": candidate_kept_mass,
                "candidate_tail_mass": candidate_tail_mass,
                "compression_attempted": bool(action["compress_reject_packet"]),
                "saving": saving,
            })

    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    acceptance_rate = correct_num_total / max(total_proposed_tokens, 1)
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)

    print(f"\n=== {run_tag} Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total proposed tokens: {total_proposed_tokens}")
    print(f"Average gamma: {sum(dynamic_gamma_hist) / max(len(dynamic_gamma_hist), 1):.2f}")
    print(f"Gamma usage: { {g: dynamic_gamma_hist.count(g) for g in sorted(set(dynamic_gamma_hist))} }")
    print(f"Compression-on ratio: {sum(dynamic_compress_hist) / max(len(dynamic_compress_hist), 1):.3f}")
    print(f"Controller switches: {controller.switch_count}")
    print(f"Controller switch rate: {controller.switch_count / max(rounds, 1):.3f}")
    print(f"Controller safe-mode count: {controller.safe_mode_count}")
    print(f"Total communication delay: {total_comm_delay:.2f}s")

    reject_rounds = max(reject_compressed_cnt + reject_full_cnt, 0)
    compression_hit_rate = (reject_compressed_cnt / reject_rounds) if reject_rounds > 0 else 0.0
    reject_downlink_saving = (1.0 - reject_down_bytes_actual / max(reject_down_bytes_full_equiv, 1)) if reject_rounds > 0 else 0.0
    avg_kept_mass = reject_kept_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_tail_mass = reject_tail_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_candidate_kept_mass = reject_candidate_kept_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0
    avg_candidate_tail_mass = reject_candidate_tail_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0

    runtime_stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "reject_compressed_cnt": reject_compressed_cnt,
        "reject_full_cnt": reject_full_cnt,
        "compression_hit_rate": compression_hit_rate,
        "reject_down_bytes_actual": reject_down_bytes_actual,
        "reject_down_bytes_full_equiv": reject_down_bytes_full_equiv,
        "reject_downlink_saving": reject_downlink_saving,
        "avg_kept_mass": avg_kept_mass,
        "avg_tail_mass": avg_tail_mass,
        "avg_candidate_kept_mass": avg_candidate_kept_mass,
        "avg_candidate_tail_mass": avg_candidate_tail_mass,
        "candidate_mass_samples": reject_candidate_mass_count,
        "avg_gamma": sum(dynamic_gamma_hist) / max(len(dynamic_gamma_hist), 1),
        "gamma_usage": _hist_counts(dynamic_gamma_hist),
        "compression_on_ratio": sum(dynamic_compress_hist) / max(len(dynamic_compress_hist), 1),
        "controller_switch_count": controller.switch_count,
        "controller_switch_rate": controller.switch_count / max(rounds, 1),
        "controller_safe_mode_count": controller.safe_mode_count,
        "controller_safe_mode_rate": controller.safe_mode_count / max(rounds, 1),
        "controller_warmup_count": controller.warmup_count,
        "controller_warmup_rate": controller.warmup_count / max(rounds, 1),
        "controller_hold_count": controller.hold_count,
        "controller_hold_rate": controller.hold_count / max(rounds, 1),
        "controller_alpha_ema_final": controller.alpha_ema,
        "controller_alpha_lcb_final": controller.alpha_lcb(),
        "controller_reject_rate_ema_final": controller.reject_rate_ema,
        "controller_early_reject_ema_final": controller.early_reject_ema,
        "controller_throughput_estimate_final": controller.throughput_estimate,
        "controller_compression_observations": controller.compression_observation_count,
        "controller_compression_history_pairs": len(controller.compression_history),
        "controller_compression_brier_mean": (
            controller.compression_brier_sum / controller.compression_observation_count
            if controller.compression_observation_count else None
        ),
        "gamma_hist": dynamic_gamma_hist,
        "topk_hist": dynamic_topk_hist,
        "mass_hist": dynamic_mass_hist,
        "compress_hist": dynamic_compress_hist,
        "reject_pos_hist": dynamic_reject_pos_hist,
        "difficulty_hist": difficulty_hist,
        "preview_time_total_s": preview_time_total,
        "preview_fresh_count": preview_fresh_count,
        "preview_cached_count": 0,
        "preview_time_per_fresh_round_ms": (
            1000.0 * preview_time_total / max(preview_fresh_count, 1)
        ),
        "controller_decision_time_total_s": controller_decision_time_total,
        "controller_decision_time_per_round_ms": (
            1000.0 * controller_decision_time_total / max(rounds, 1)
        ),
    }
    runtime_stats.update(controller_link.stats())
    runtime_stats.update(_latency_stats(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_stats(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_stats(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_stats(round_llm_latency_hist, "round_llm_latency"))
    runtime_stats.update(_latency_samples(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_samples(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_samples(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_samples(round_llm_latency_hist, "round_llm_latency"))

    return generated, throughput, total_time, acceptance_rate, total_comm_delay, total_dup_bytes, rounds, total_slm_time, total_llm_time, runtime_stats


def generate_async_dynamic_DSSD(uav_node: UAVNode, bs_node: BSNode, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace, run_tag: str = "Async Dynamic DSSD"):
    """
    Async DSSD: 在 Dynamic DSSD 基础上增加一阶段异步 shadow drafting。

    核心思想：
      - 当前轮在 BS 端验证时，UAV 端基于 "accept-all + 本地锚点 token" 的乐观假设，
        异步生成下一轮 shadow draft。
      - 当且仅当 BS 返回的 x_{gamma+1} 与本地锚点 token 一致时，shadow 才被复用。
      - 若发生 reject 或锚点不一致，则丢弃 shadow，不影响正确性。
    """
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]
    link_replay = TimeVaryingLinkReplay.from_args(args)
    controller_link = ControllerLinkTelemetry(link_replay, args)

    controller = DynamicDSSDController(args)
    total_comm_delay = 0.0
    total_slm_time = 0.0
    total_llm_time = 0.0
    rounds = 0
    correct_num_total = 0
    reject_num_total = 0
    total_dup_bytes = 0
    total_proposed_tokens = 0

    reject_compressed_cnt = 0
    reject_full_cnt = 0
    accept_all_rounds = 0
    reject_down_bytes_actual = 0
    reject_down_bytes_full_equiv = 0
    reject_kept_mass_sum = 0.0
    reject_tail_mass_sum = 0.0
    reject_candidate_kept_mass_sum = 0.0
    reject_candidate_tail_mass_sum = 0.0
    reject_candidate_mass_count = 0

    dynamic_gamma_hist = []
    dynamic_topk_hist = []
    dynamic_mass_hist = []
    dynamic_compress_hist = []
    dynamic_reject_pos_hist = []
    difficulty_hist = []
    round_total_latency_hist = []
    round_comm_latency_hist = []
    round_slm_latency_hist = []
    round_llm_latency_hist = []
    preview_time_total = 0.0
    preview_fresh_count = 0
    preview_cached_count = 0
    controller_decision_time_total = 0.0

    async_shadow_hits = 0
    async_shadow_misses = 0
    async_shadow_discard_reject = 0
    async_shadow_discard_mismatch = 0
    async_shadow_discard_gamma_change = 0
    async_shadow_discard_prefix_mismatch = 0
    async_shadow_discard_generation_end = 0
    async_shadow_partial_extensions = 0
    async_shadow_eligible_rounds = 0
    async_shadow_blocked_overlap = 0
    async_shadow_blocked_benefit = 0
    async_shadow_launches = 0
    async_shadow_total_time = 0.0
    async_shadow_wait_time = 0.0
    async_shadow_wasted_compute_time = 0.0
    async_shadow_reused_compute_time = 0.0
    async_shadow_saved_draft_time = 0.0
    async_shadow_reused_gamma = []
    async_shadow_launch_gamma = []
    async_shadow_tokens_computed = 0
    async_shadow_cancelled = 0
    async_shadow_accept_all_launches = 0
    async_shadow_anchor_matches = 0
    async_shadow_predicted_reuse = []
    async_shadow_expected_benefit = []
    shadow_compute_per_token_ema = 0.0
    observed_rounds_by_gamma = {}
    observed_accept_all_by_gamma = {}

    prefix = input_ids
    torch.manual_seed(args.seed)
    shadow_cache = None

    # 不同 GPU 上用线程最稳；若同一 GPU，则仍可运行，但收益可能很小。
    with ThreadPoolExecutor(max_workers=1) as executor:
        with tqdm(total=max_total_len, desc=f"{run_tag}: Async Dynamic Split Speculative Decoding") as pbar:
            pbar.update(prefix.shape[1])
            initial_len = prefix.shape[1]
            start_time = time.time()

            while prefix.shape[1] < max_total_len:
                round_start = time.time()
                old_len = prefix.shape[1]
                if (
                    shadow_cache is not None
                    and shadow_cache.get("prefix_len") != int(prefix.shape[1])
                ):
                    async_shadow_misses += 1
                    async_shadow_discard_prefix_mismatch += 1
                    async_shadow_wasted_compute_time += float(
                        shadow_cache.get("shadow_compute_time_s", 0.0)
                    )
                    shadow_cache = None

                if shadow_cache is not None:
                    difficulty_info = uav_node.difficulty_from_probs(
                        shadow_cache["q_probs"][0]
                    )
                    preview_time = 0.0
                    preview_cached_count += 1
                else:
                    preview_start = time.time()
                    difficulty_info = uav_node.preview_difficulty(prefix)
                    preview_time = time.time() - preview_start
                    preview_time_total += preview_time
                    preview_fresh_count += 1
                difficulty = float(difficulty_info["difficulty"])
                difficulty_hist.append(difficulty)

                state = {
                    "difficulty": difficulty,
                    "jitter": 0.0,
                    "vocab_size": int(getattr(tokenizer, 'vocab_size', 32000)),
                    "tx_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="down"),
                    "up_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="up"),
                    "down_delay_fn": lambda size_B: controller_link.delay_bytes(int(size_B), direction="down"),
                }
                decision_start = time.perf_counter()
                action = controller.choose_action(state)
                controller_decision_time_total += time.perf_counter() - decision_start

                gamma_t = int(action["gamma"])
                remain = max_total_len - prefix.shape[1]
                gamma_t = min(gamma_t, max(1, remain))

                uav_node.args.gamma = gamma_t
                bs_node.args.gamma = gamma_t
                uav_node.args.compress_reject_packet = bool(action["compress_reject_packet"])
                bs_node.args.compress_reject_packet = bool(action["compress_reject_packet"])
                uav_node.args.reject_topk = int(action["reject_topk"])
                bs_node.args.reject_topk = int(action["reject_topk"])
                uav_node.args.reject_mass_threshold = float(action["reject_mass_threshold"])
                bs_node.args.reject_mass_threshold = float(action["reject_mass_threshold"])
                uav_node.args.reject_packet_dtype = action["reject_packet_dtype"]
                bs_node.args.reject_packet_dtype = action["reject_packet_dtype"]

                dynamic_gamma_hist.append(gamma_t)
                dynamic_topk_hist.append(int(action["reject_topk"]))
                dynamic_mass_hist.append(float(action["reject_mass_threshold"]))
                dynamic_compress_hist.append(int(bool(action["compress_reject_packet"])))

                rounds += 1
                total_proposed_tokens += gamma_t

                # 1) 当前轮 draft：优先尝试复用上一轮异步生成的 shadow
                shadow_used = False
                controller_slm_time = None
                counterfactual_saved_time = 0.0
                t_slm_start = time.time()
                if shadow_cache is not None:
                    shadow_compute_time = float(shadow_cache.get("shadow_compute_time_s", 0.0))
                    shadow_generated_gamma = max(int(shadow_cache.get("shadow_gamma", gamma_t)), 1)
                    if gamma_t > shadow_generated_gamma:
                        async_shadow_partial_extensions += 1
                    x_draft, q_values, q_probs, dup_bytes = uav_node.materialize_shadow_for_gamma(shadow_cache, gamma_t)
                    shadow_used = True
                    async_shadow_hits += 1
                    async_shadow_reused_gamma.append(gamma_t)
                    used_fraction = (
                        min(gamma_t, shadow_generated_gamma)
                        / (shadow_generated_gamma + 1)
                    )
                    async_shadow_reused_compute_time += shadow_compute_time * used_fraction
                    async_shadow_wasted_compute_time += shadow_compute_time * (1.0 - used_fraction)
                    counterfactual_saved_time = shadow_compute_time * used_fraction
                    shadow_cache = None
                else:
                    x_draft, q_values, q_probs, dup_bytes = uav_node.draft_DSSD(
                        prefix, gamma_t, first_logits=difficulty_info["first_logits"]
                    )
                round_slm = preview_time + (time.time() - t_slm_start)
                controller_slm_time = round_slm + counterfactual_saved_time
                if shadow_used:
                    async_shadow_saved_draft_time += counterfactual_saved_time
                total_dup_bytes += dup_bytes
                total_slm_time += round_slm

                # 2) 上行开始时即启动 shadow。本地已有本轮 draft packet，
                # 因此 shadow 可与上行、BS verification 及下行反馈整个窗口重叠。
                t_up = link_replay.delay_bytes(dup_bytes, direction="up")
                total_comm_delay += t_up
                future_shadow = None
                shadow_cancel_event = None
                shadow_gamma = 0
                launch_mode = str(getattr(args, "dynamic_async_launch_mode", "adaptive"))

                # The controller is stabilized by residency, so the current gamma is
                # the least speculative estimate of the next-round requirement.
                remain_after_accept = max_total_len - min(max_total_len, x_draft.shape[1] + 1)
                shadow_cap = int(getattr(args, "async_shadow_gamma_cap", max(getattr(controller, "gamma_set", [gamma_t]))))
                base_shadow_gamma = (
                    min(shadow_cap, gamma_t, remain_after_accept)
                    if remain_after_accept > 0 else 0
                )

                alpha_pred = controller.predict_alpha(difficulty, 0.0)
                gamma_rounds = int(observed_rounds_by_gamma.get(gamma_t, 0))
                gamma_accept_all = int(observed_accept_all_by_gamma.get(gamma_t, 0))
                empirical_accept_all = (gamma_accept_all + 1.0) / (gamma_rounds + 2.0)
                modeled_accept_all = alpha_pred ** gamma_t
                p_accept_all = 0.5 * empirical_accept_all + 0.5 * modeled_accept_all
                if async_shadow_accept_all_launches > 0:
                    empirical_anchor_match = (
                        (async_shadow_anchor_matches + 1.0)
                        / (async_shadow_accept_all_launches + 2.0)
                    )
                    p_anchor_match = 0.5 * empirical_anchor_match + 0.5 * alpha_pred
                else:
                    p_anchor_match = alpha_pred
                p_reuse = max(0.0, min(1.0, p_accept_all * p_anchor_match))

                observed_slm_per_token = controller_slm_time / max(gamma_t, 1)
                slm_per_token = max(
                    controller.t_slm_per_tok,
                    observed_slm_per_token,
                    shadow_compute_per_token_ema,
                )
                shadow_compute_est = slm_per_token * (base_shadow_gamma + 1)
                overlap_est = max(
                    0.0,
                    t_up
                    + controller.estimate_t_llm(gamma_t)
                    + state["down_delay_fn"](controller.down_bytes_accept()),
                )
                saved_time = min(shadow_compute_est, overlap_est)
                # For throughput, failed shadow work matters only when it extends
                # beyond the remote verify/feedback window. Full wasted local
                # compute is still recorded separately for resource/energy analysis.
                unhidden_overhang = max(0.0, shadow_compute_est - overlap_est)
                expected_benefit = (
                    p_reuse * saved_time
                    - (1.0 - p_reuse) * unhidden_overhang
                )
                eligible = (
                    args.async_shadow_enabled
                    and shadow_cache is None
                    and base_shadow_gamma > 0
                )
                overlap_margin = float(
                    getattr(args, "async_shadow_overlap_margin", 1.0)
                )
                fits_overlap = shadow_compute_est <= overlap_est * overlap_margin
                if launch_mode == "always":
                    should_launch_shadow = eligible
                else:
                    should_launch_shadow = (
                        eligible and expected_benefit > 0.0 and fits_overlap
                    )
                if eligible:
                    async_shadow_eligible_rounds += 1
                    if launch_mode != "always":
                        if not fits_overlap:
                            async_shadow_blocked_overlap += 1
                        elif expected_benefit <= 0.0:
                            async_shadow_blocked_benefit += 1

                if should_launch_shadow:
                    shadow_gamma = base_shadow_gamma
                    async_shadow_launches += 1
                    async_shadow_launch_gamma.append(shadow_gamma)
                    async_shadow_predicted_reuse.append(p_reuse)
                    async_shadow_expected_benefit.append(expected_benefit)
                    shadow_cancel_event = threading.Event()
                    future_shadow = executor.submit(
                        uav_node.prepare_shadow_candidate,
                        x_draft.detach().clone(), shadow_gamma, shadow_cancel_event,
                    )

                time.sleep(t_up)

                # 3) BS verify
                t_llm_start = time.time()
                j, flag, reject_packet, xj, correct_num, reject_num = bs_node.verify_DSSD(x_draft, q_values, gamma_t)
                round_llm = time.time() - t_llm_start
                total_llm_time += round_llm
                correct_num_total += correct_num
                reject_num_total += reject_num

                prefix_len = prefix.shape[1]
                kept_mass = 1.0
                tail_mass = 0.0
                candidate_kept_mass = 1.0
                candidate_tail_mass = 0.0
                saving = 0.0

                if flag == 1:
                    accept_all_rounds += 1
                    new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                    if new_prefix.shape[1] > max_total_len:
                        new_prefix = new_prefix[:, :max_total_len]
                    down_bytes = 4 + xj.numel() * 4
                    dynamic_reject_pos_hist.append(0)
                else:
                    dynamic_reject_pos_hist.append(int(j))
                    if reject_packet["mode"] == "compressed":
                        reject_compressed_cnt += 1
                    else:
                        reject_full_cnt += 1
                    kept_mass = float(reject_packet.get("kept_mass", 1.0))
                    tail_mass = float(reject_packet.get("tail_mass", 0.0))
                    candidate_kept_mass = float(reject_packet.get("candidate_kept_mass", kept_mass))
                    candidate_tail_mass = float(reject_packet.get("candidate_tail_mass", tail_mass))
                    reject_kept_mass_sum += kept_mass
                    reject_tail_mass_sum += tail_mass
                    if bool(action["compress_reject_packet"]):
                        reject_candidate_kept_mass_sum += candidate_kept_mass
                        reject_candidate_tail_mass_sum += candidate_tail_mass
                        reject_candidate_mass_count += 1

                    packet_bytes_actual = reject_packet_nbytes(reject_packet)
                    packet_bytes_full = full_packet_equivalent_nbytes(reject_packet)
                    reject_down_bytes_actual += packet_bytes_actual
                    reject_down_bytes_full_equiv += packet_bytes_full
                    saving = 1.0 - packet_bytes_actual / max(packet_bytes_full, 1)

                    xj_prime = uav_node.resample_DSSD(j, reject_packet, q_probs)
                    x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                    new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
                    down_bytes = 4 + packet_bytes_actual

                observed_rounds_by_gamma[gamma_t] = gamma_rounds + 1
                if flag == 1:
                    observed_accept_all_by_gamma[gamma_t] = gamma_accept_all + 1

                # 5) 下行
                t_down = link_replay.delay_bytes(down_bytes, direction="down")
                total_comm_delay += t_down
                time.sleep(t_down)
                controller_link.observe_completed_round()

                # 6) 处理异步 shadow 的提交/丢弃
                if future_shadow is not None:
                    if flag == 0 and shadow_cancel_event is not None:
                        shadow_cancel_event.set()
                    t_shadow_wait = time.time()
                    shadow_result = future_shadow.result()
                    async_shadow_wait_time += time.time() - t_shadow_wait
                    shadow_tokens = int(shadow_result.get("shadow_tokens_computed", 0))
                    shadow_compute_time = float(shadow_result.get("shadow_compute_time_s", 0.0))
                    async_shadow_tokens_computed += shadow_tokens
                    async_shadow_total_time += shadow_compute_time
                    observed_shadow_per_token = shadow_compute_time / max(shadow_tokens + 1, 1)
                    if shadow_compute_per_token_ema <= 0.0:
                        shadow_compute_per_token_ema = observed_shadow_per_token
                    else:
                        shadow_compute_per_token_ema = (
                            0.8 * shadow_compute_per_token_ema
                            + 0.2 * observed_shadow_per_token
                        )
                    if flag == 0:
                        async_shadow_discard_reject += 1
                        async_shadow_cancelled += int(bool(shadow_result.get("cancelled", False)))
                        async_shadow_wasted_compute_time += float(shadow_result.get("shadow_compute_time_s", 0.0))
                        shadow_cache = None
                    elif shadow_result.get("cancelled", False):
                        async_shadow_misses += 1
                        async_shadow_cancelled += 1
                        async_shadow_wasted_compute_time += float(shadow_result.get("shadow_compute_time_s", 0.0))
                        shadow_cache = None
                    else:
                        async_shadow_accept_all_launches += 1
                        returned_anchor = int(xj.item())
                        if returned_anchor == int(shadow_result["anchor_token_id"]):
                            async_shadow_anchor_matches += 1
                            shadow_cache = dict(shadow_result)
                            shadow_cache["prefix_len"] = int(new_prefix.shape[1])
                        else:
                            async_shadow_discard_mismatch += 1
                            async_shadow_wasted_compute_time += float(shadow_result.get("shadow_compute_time_s", 0.0))
                            shadow_cache = None

                prefix = new_prefix
                new_len = prefix.shape[1]
                pbar.update(new_len - old_len)
                round_total_latency_hist.append(time.time() - round_start)
                round_comm_latency_hist.append(t_up + t_down)
                round_slm_latency_hist.append(round_slm)
                round_llm_latency_hist.append(round_llm)

                controller.update({
                    "elapsed_s": time.time() - start_time,
                    "generated_tokens": new_len - initial_len,
                    "reject_topk": action["reject_topk"],
                    "reject_mass_threshold": action["reject_mass_threshold"],
                    "compressed_feedback": flag == 0 and reject_packet["mode"] == "compressed",
                    "gamma": gamma_t,
                    "correct_num": correct_num,
                    "flag": flag,
                    "j": int(j),
                    "difficulty": difficulty,
                    "t_slm": controller_slm_time,
                    "t_llm": round_llm,
                    "kept_mass": kept_mass,
                    "tail_mass": tail_mass,
                    "candidate_kept_mass": candidate_kept_mass,
                    "candidate_tail_mass": candidate_tail_mass,
                    "compression_attempted": bool(action["compress_reject_packet"]),
                    "saving": saving,
                })

    if shadow_cache is not None:
        async_shadow_misses += 1
        async_shadow_discard_generation_end += 1
        async_shadow_wasted_compute_time += float(shadow_cache.get("shadow_compute_time_s", 0.0))

    # A shadow miss is any launched candidate that is not eventually reused.
    # Reject and anchor-mismatch counters provide the mutually informative causes.
    async_shadow_misses_total = max(async_shadow_launches - async_shadow_hits, 0)

    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    acceptance_rate = correct_num_total / max(total_proposed_tokens, 1)
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)

    print(f"\n=== {run_tag} Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total proposed tokens: {total_proposed_tokens}")
    print(f"Average gamma: {sum(dynamic_gamma_hist) / max(len(dynamic_gamma_hist), 1):.2f}")
    print(f"Gamma usage: { {g: dynamic_gamma_hist.count(g) for g in sorted(set(dynamic_gamma_hist))} }")
    print(f"Compression-on ratio: {sum(dynamic_compress_hist) / max(len(dynamic_compress_hist), 1):.3f}")
    print(f"Controller switches: {controller.switch_count}")
    print(f"Controller switch rate: {controller.switch_count / max(rounds, 1):.3f}")
    print(f"Controller safe-mode count: {controller.safe_mode_count}")
    print(f"Async shadow launches: {async_shadow_launches}")
    print(f"Async shadow hits: {async_shadow_hits}")
    print(f"Async shadow misses: {async_shadow_misses_total}")
    print(f"Async shadow hit rate: {async_shadow_hits / max(async_shadow_launches, 1):.3f}")
    print(f"Async discard(reject): {async_shadow_discard_reject}")
    print(f"Async discard(mismatch): {async_shadow_discard_mismatch}")
    print(f"Async discard(gamma change): {async_shadow_discard_gamma_change}")
    print(f"Async discard(prefix mismatch): {async_shadow_discard_prefix_mismatch}")
    print(f"Async discard(generation end): {async_shadow_discard_generation_end}")
    print(f"Async partial extensions: {async_shadow_partial_extensions}")
    print(
        "Async launch eligible/blocked-overlap/blocked-benefit: "
        f"{async_shadow_eligible_rounds}/{async_shadow_blocked_overlap}/"
        f"{async_shadow_blocked_benefit}"
    )
    print(f"Async accept-all launches: {async_shadow_accept_all_launches}")
    print(f"Async anchor matches: {async_shadow_anchor_matches}")
    print(f"Total communication delay: {total_comm_delay:.2f}s")

    reject_rounds = max(reject_compressed_cnt + reject_full_cnt, 0)
    compression_hit_rate = (reject_compressed_cnt / reject_rounds) if reject_rounds > 0 else 0.0
    reject_downlink_saving = (1.0 - reject_down_bytes_actual / max(reject_down_bytes_full_equiv, 1)) if reject_rounds > 0 else 0.0
    avg_kept_mass = reject_kept_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_tail_mass = reject_tail_mass_sum / reject_rounds if reject_rounds > 0 else 0.0
    avg_candidate_kept_mass = reject_candidate_kept_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0
    avg_candidate_tail_mass = reject_candidate_tail_mass_sum / reject_candidate_mass_count if reject_candidate_mass_count > 0 else 0.0

    runtime_stats = {
        "accept_all_rounds": accept_all_rounds,
        "reject_rounds": reject_rounds,
        "reject_compressed_cnt": reject_compressed_cnt,
        "reject_full_cnt": reject_full_cnt,
        "compression_hit_rate": compression_hit_rate,
        "reject_down_bytes_actual": reject_down_bytes_actual,
        "reject_down_bytes_full_equiv": reject_down_bytes_full_equiv,
        "reject_downlink_saving": reject_downlink_saving,
        "avg_kept_mass": avg_kept_mass,
        "avg_tail_mass": avg_tail_mass,
        "avg_candidate_kept_mass": avg_candidate_kept_mass,
        "avg_candidate_tail_mass": avg_candidate_tail_mass,
        "candidate_mass_samples": reject_candidate_mass_count,
        "avg_gamma": sum(dynamic_gamma_hist) / max(len(dynamic_gamma_hist), 1),
        "gamma_usage": _hist_counts(dynamic_gamma_hist),
        "compression_on_ratio": sum(dynamic_compress_hist) / max(len(dynamic_compress_hist), 1),
        "controller_switch_count": controller.switch_count,
        "controller_switch_rate": controller.switch_count / max(rounds, 1),
        "controller_safe_mode_count": controller.safe_mode_count,
        "controller_safe_mode_rate": controller.safe_mode_count / max(rounds, 1),
        "controller_warmup_count": controller.warmup_count,
        "controller_warmup_rate": controller.warmup_count / max(rounds, 1),
        "controller_hold_count": controller.hold_count,
        "controller_hold_rate": controller.hold_count / max(rounds, 1),
        "controller_alpha_ema_final": controller.alpha_ema,
        "controller_alpha_lcb_final": controller.alpha_lcb(),
        "controller_reject_rate_ema_final": controller.reject_rate_ema,
        "controller_early_reject_ema_final": controller.early_reject_ema,
        "controller_throughput_estimate_final": controller.throughput_estimate,
        "controller_compression_observations": controller.compression_observation_count,
        "controller_compression_history_pairs": len(controller.compression_history),
        "controller_compression_brier_mean": (
            controller.compression_brier_sum / controller.compression_observation_count
            if controller.compression_observation_count else None
        ),
        "gamma_hist": dynamic_gamma_hist,
        "topk_hist": dynamic_topk_hist,
        "mass_hist": dynamic_mass_hist,
        "compress_hist": dynamic_compress_hist,
        "reject_pos_hist": dynamic_reject_pos_hist,
        "difficulty_hist": difficulty_hist,
        "preview_time_total_s": preview_time_total,
        "preview_fresh_count": preview_fresh_count,
        "preview_cached_count": preview_cached_count,
        "preview_time_per_fresh_round_ms": (
            1000.0 * preview_time_total / max(preview_fresh_count, 1)
        ),
        "controller_decision_time_total_s": controller_decision_time_total,
        "controller_decision_time_per_round_ms": (
            1000.0 * controller_decision_time_total / max(rounds, 1)
        ),
        "async_shadow_launches": async_shadow_launches,
        "async_shadow_launch_rate": async_shadow_launches / max(rounds, 1),
        "async_shadow_hits": async_shadow_hits,
        "async_shadow_hit_rate": async_shadow_hits / max(async_shadow_launches, 1),
        "async_shadow_misses": async_shadow_misses_total,
        "async_shadow_discard_reject": async_shadow_discard_reject,
        "async_shadow_discard_mismatch": async_shadow_discard_mismatch,
        "async_shadow_discard_gamma_change": async_shadow_discard_gamma_change,
        "async_shadow_discard_prefix_mismatch": async_shadow_discard_prefix_mismatch,
        "async_shadow_discard_generation_end": async_shadow_discard_generation_end,
        "async_shadow_partial_extensions": async_shadow_partial_extensions,
        "async_shadow_eligible_rounds": async_shadow_eligible_rounds,
        "async_shadow_blocked_overlap": async_shadow_blocked_overlap,
        "async_shadow_blocked_benefit": async_shadow_blocked_benefit,
        "async_shadow_cancelled": async_shadow_cancelled,
        "async_shadow_accept_all_launches": async_shadow_accept_all_launches,
        "async_shadow_anchor_matches": async_shadow_anchor_matches,
        "async_shadow_anchor_match_rate": (
            async_shadow_anchor_matches / max(async_shadow_accept_all_launches, 1)
        ),
        "async_shadow_tokens_computed": async_shadow_tokens_computed,
        "async_shadow_predicted_reuse_mean": (
            sum(async_shadow_predicted_reuse) / max(len(async_shadow_predicted_reuse), 1)
        ),
        "async_shadow_expected_benefit_mean_s": (
            sum(async_shadow_expected_benefit) / max(len(async_shadow_expected_benefit), 1)
        ),
        "async_shadow_total_time_s": async_shadow_total_time,
        "async_shadow_wait_time_s": async_shadow_wait_time,
        "async_shadow_wasted_compute_time_s": async_shadow_wasted_compute_time,
        "async_shadow_reused_compute_time_s": async_shadow_reused_compute_time,
        "async_shadow_saved_draft_time_s": async_shadow_saved_draft_time,
        "async_shadow_compute_per_token_ema_s": shadow_compute_per_token_ema,
        "async_shadow_anchor_mode": str(
            getattr(args, "async_shadow_anchor_mode", "sample")
        ),
        "async_shadow_overlap_margin": float(
            getattr(args, "async_shadow_overlap_margin", 1.0)
        ),
        "async_shadow_reused_gamma": async_shadow_reused_gamma,
        "async_shadow_reused_gamma_usage": _hist_counts(async_shadow_reused_gamma),
        "async_shadow_launch_gamma_usage": _hist_counts(async_shadow_launch_gamma),
    }
    runtime_stats.update(controller_link.stats())
    runtime_stats.update(_latency_stats(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_stats(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_stats(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_stats(round_llm_latency_hist, "round_llm_latency"))
    runtime_stats.update(_latency_samples(round_total_latency_hist, "round_total_latency"))
    runtime_stats.update(_latency_samples(round_comm_latency_hist, "round_comm_latency"))
    runtime_stats.update(_latency_samples(round_slm_latency_hist, "round_slm_latency"))
    runtime_stats.update(_latency_samples(round_llm_latency_hist, "round_llm_latency"))

    return generated, throughput, total_time, acceptance_rate, total_comm_delay, total_dup_bytes, rounds, total_slm_time, total_llm_time, runtime_stats


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--input', type=str, default="Hello, how are you?")
    parser.add_argument('--workload', type=str, default='', help='workload or dataset name for CSV logging')
    parser.add_argument('--prompt_id', type=str, default='', help='prompt identifier for CSV logging')
    parser.add_argument('--run_tag', type=str, default='', help='optional run tag for CSV logging')
    parser.add_argument('--draft_model_name', type=str, default="models/vicuna-68m")
    parser.add_argument('--target_model_name', type=str, default="models/vicuna-7b-v1.5")  
    parser.add_argument('--max_len', type=int, default=80) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--rtt', type=float, default=0.02)
    parser.add_argument('--bandwidth', type=float, default=100, help='Bandwidth in Mbps')
    parser.add_argument('--link_trace_path', type=str, default='', help='Optional JSON/JSONL time-varying link trace for replay')
    parser.add_argument('--link_trace_speedup', type=float, default=1.0,
                        help='Replay trace time at this multiple of wall-clock speed')
    parser.add_argument(
        '--controller_link_mode',
        choices=['instantaneous', 'causal_previous_round'],
        default='causal_previous_round',
        help='link state exposed to the online controller',
    )
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--csv_path', type=str, default="results.csv")
    parser.add_argument('--device_1', type=str, default="cpu")
    parser.add_argument('--device_2', type=str, default="cuda:0")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--use_dist_summary', action='store_true', help='upload compressed distribution instead of raw logits')
    parser.add_argument('--no_cache', action='store_true', help='disable Δ-prompt cache (ablation)')
    parser.add_argument('--compress_reject_packet', action='store_true',
                        help='compress reject downlink packet in DSSD')
    parser.add_argument('--reject_topk', type=int, default=64,
                        help='top-k entries kept in compressed reject packet')
    parser.add_argument('--reject_mass_threshold', type=float, default=0.995,
                        help='fallback to full packet when top-k cumulative mass is below this threshold')
    parser.add_argument('--reject_packet_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='dtype used for compressed reject packet probs')
    parser.add_argument('--reject_min_prob', type=float, default=0.0,
                        help='fallback to full packet when the kth kept probability is too small')
    parser.set_defaults(compress_reject_packet=True)

    # Dynamic DSSD controller
    parser.add_argument('--enable_dynamic_dssd', action='store_true', help='run Dynamic DSSD on top of Optimized DSSD')
    parser.add_argument('--dynamic_joint_packet_control', action='store_true', help='jointly adapt gamma and reject packet policy')
    parser.add_argument('--dynamic_gamma_set', type=str, default='1,2,3,4,6,8')
    parser.add_argument('--dynamic_min_gamma', type=int, default=1)
    parser.add_argument('--dynamic_reject_topk_set', type=str, default='16,32,64,128')
    parser.add_argument('--dynamic_reject_mass_set', type=str, default='0.98,0.99,0.995')
    parser.add_argument('--dynamic_beta', type=float, default=0.2)
    parser.add_argument('--dynamic_score_mode', choices=['ratio', 'difference'], default='ratio')
    parser.add_argument('--dynamic_compression_estimator', choices=['legacy', 'history'], default='legacy')
    parser.add_argument('--dynamic_acceptance_predictor', choices=['legacy', 'lcb'], default='legacy')
    parser.add_argument('--dynamic_z_alpha', type=float, default=0.7)
    parser.add_argument('--dynamic_alpha_weight_history', type=float, default=0.65)
    parser.add_argument('--dynamic_alpha_weight_difficulty', type=float, default=0.20)
    parser.add_argument('--dynamic_alpha_weight_early_reject', type=float, default=0.15)
    parser.add_argument('--dynamic_cmp_weight_mass', type=float, default=0.45)
    parser.add_argument('--dynamic_cmp_weight_saving', type=float, default=0.25)
    parser.add_argument('--dynamic_cmp_weight_topk', type=float, default=0.20)
    parser.add_argument('--dynamic_cmp_weight_easiness', type=float, default=0.10)
    parser.add_argument('--dynamic_cmp_threshold_penalty', type=float, default=0.15)
    parser.add_argument('--dynamic_cmp_k_ref', type=float, default=64.0)
    parser.add_argument('--dynamic_cmp_tau0', type=float, default=0.98)
    parser.add_argument('--dynamic_cmp_delta_tau', type=float, default=0.02)
    parser.add_argument('--dynamic_lambda_sw', type=float, default=0.02)
    parser.add_argument('--dynamic_lambda_cmp', type=float, default=0.02)
    parser.add_argument('--dynamic_lambda_tail', type=float, default=0.04)
    parser.add_argument('--dynamic_hysteresis', type=float, default=0.01)
    parser.add_argument('--dynamic_min_residency', type=int, default=2)
    parser.add_argument('--dynamic_alpha_panic', type=float, default=0.18)
    parser.add_argument('--dynamic_safe_gamma', type=int, default=1)
    parser.add_argument('--dynamic_downlink_budget_frac', type=float, default=0.45)
    parser.add_argument('--dynamic_warmup_rounds', type=int, default=6)
    parser.add_argument('--dynamic_panic_reject_rate', type=float, default=0.80)
    parser.add_argument('--dynamic_panic_early_reject', type=float, default=0.70)
    parser.add_argument('--dynamic_debug', action='store_true', help='print controller safe-mode/switch decisions each round')
    parser.set_defaults(enable_dynamic_dssd=True)
    parser.add_argument('--async_shadow_enabled', action='store_true', help='enable optimistic one-window async shadow drafting on top of Dynamic DSSD')
    parser.add_argument('--async_shadow_gamma_cap', type=int, default=8, help='max shadow gamma drafted asynchronously for next round reuse')
    parser.add_argument('--async_shadow_anchor_mode', type=str, default='sample', choices=['sample', 'map'],
                        help='stochastic or draft-top-1 anchor prediction for shadow reuse')
    parser.add_argument('--async_shadow_overlap_margin', type=float, default=1.0,
                        help='adaptive launch requires estimated shadow time <= margin * overlap window')
    parser.add_argument('--dynamic_async_launch_mode', type=str, default='adaptive', choices=['adaptive', 'always'],
                        help='adaptive expected-benefit gating or always-launch diagnostic')
    parser.set_defaults(async_shadow_enabled=True)
    parser.set_defaults(dynamic_joint_packet_control=True)
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_arguments()
    recorder = Recorder(args.csv_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理未使用的显存
    device_1 = torch.device(args.device_1) 
    device_2 = torch.device(args.device_2)
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)
    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    
    # 创建多组节点，分别对应 Vanilla DSSD / Optimized DSSD / DSD
    vanilla_args = copy.deepcopy(args)
    vanilla_args.compress_reject_packet = False

    optimized_args = copy.deepcopy(args)
    optimized_args.compress_reject_packet = True

    dsd_args = copy.deepcopy(args)

    uav_node_vanilla = UAVNode(draft_model, device_1, vanilla_args)
    bs_node_vanilla = BSNode(target_model, device_2, vanilla_args)

    uav_node_opt = UAVNode(draft_model, device_1, optimized_args)
    bs_node_opt = BSNode(target_model, device_2, optimized_args)

    dynamic_args = copy.deepcopy(args)

    uav_node_dsd = UAVNode(draft_model, device_1, dsd_args)
    bs_node_dsd = BSNode(target_model, device_2, dsd_args)

    uav_node_dynamic = UAVNode(draft_model, device_1, dynamic_args)
    bs_node_dynamic = BSNode(target_model, device_2, dynamic_args)

    # 执行 Vanilla DSSD（不启用 reject packet 压缩）
    (
        vanilla_dssd_generated,
        vanilla_dssd_throughput,
        vanilla_dssd_time,
        vanilla_dssd_acceptance_rate,
        vanilla_dssd_total_comm,
        vanilla_dssd_total_dup_bytes,
        vanilla_dssd_rounds,
        vanilla_dssd_total_slm,
        vanilla_dssd_total_llm,
        vanilla_dssd_reject_stats,
    ) = generate_DSSD(
        uav_node_vanilla, bs_node_vanilla, input_ids, tokenizer, vanilla_args,
        run_tag="Vanilla DSSD", show_compression_stats=False
    )

    # 执行 Optimized DSSD（启用 reject packet 压缩）
    (
        opt_dssd_generated,
        opt_dssd_throughput,
        opt_dssd_time,
        opt_dssd_acceptance_rate,
        opt_dssd_total_comm,
        opt_dssd_total_dup_bytes,
        opt_dssd_rounds,
        opt_dssd_total_slm,
        opt_dssd_total_llm,
        opt_dssd_reject_stats,
    ) = generate_DSSD(
        uav_node_opt, bs_node_opt, input_ids, tokenizer, optimized_args,
        run_tag="Optimized DSSD", show_compression_stats=True
    )

    # 执行 Dynamic DSSD（无 async shadow）
    if args.enable_dynamic_dssd:
        dynamic_only_args = copy.deepcopy(dynamic_args)
        dynamic_only_args.async_shadow_enabled = False
        (
            dynamic_dssd_generated,
            dynamic_dssd_throughput,
            dynamic_dssd_time,
            dynamic_dssd_acceptance_rate,
            dynamic_dssd_total_comm,
            dynamic_dssd_total_dup_bytes,
            dynamic_dssd_rounds,
            dynamic_dssd_total_slm,
            dynamic_dssd_total_llm,
            dynamic_dssd_stats,
        ) = generate_dynamic_DSSD(
            uav_node_dynamic, bs_node_dynamic, input_ids, tokenizer, dynamic_only_args,
            run_tag="Dynamic DSSD"
        )
    else:
        dynamic_dssd_generated = ""
        dynamic_dssd_throughput = 0.0
        dynamic_dssd_time = 0.0
        dynamic_dssd_acceptance_rate = 0.0
        dynamic_dssd_total_comm = 0.0
        dynamic_dssd_total_dup_bytes = 0
        dynamic_dssd_rounds = 0
        dynamic_dssd_total_slm = 0.0
        dynamic_dssd_total_llm = 0.0
        dynamic_dssd_stats = {}

    # 执行 Async Dynamic DSSD（在 Dynamic DSSD 基础上启用 async shadow）
    if args.enable_dynamic_dssd and args.async_shadow_enabled:
        async_dynamic_args = copy.deepcopy(dynamic_args)
        async_dynamic_args.async_shadow_enabled = True
        uav_node_async_dynamic = UAVNode(draft_model, device_1, async_dynamic_args)
        bs_node_async_dynamic = BSNode(target_model, device_2, async_dynamic_args)
        (
            async_dynamic_dssd_generated,
            async_dynamic_dssd_throughput,
            async_dynamic_dssd_time,
            async_dynamic_dssd_acceptance_rate,
            async_dynamic_dssd_total_comm,
            async_dynamic_dssd_total_dup_bytes,
            async_dynamic_dssd_rounds,
            async_dynamic_dssd_total_slm,
            async_dynamic_dssd_total_llm,
            async_dynamic_dssd_stats,
        ) = generate_async_dynamic_DSSD(
            uav_node_async_dynamic, bs_node_async_dynamic, input_ids, tokenizer, async_dynamic_args,
            run_tag="Async Dynamic DSSD"
        )
    else:
        async_dynamic_dssd_generated = ""
        async_dynamic_dssd_throughput = 0.0
        async_dynamic_dssd_time = 0.0
        async_dynamic_dssd_acceptance_rate = 0.0
        async_dynamic_dssd_total_comm = 0.0
        async_dynamic_dssd_total_dup_bytes = 0
        async_dynamic_dssd_rounds = 0
        async_dynamic_dssd_total_slm = 0.0
        async_dynamic_dssd_total_llm = 0.0
        async_dynamic_dssd_stats = {}

    # 执行分布式投机采样（DSD）
    (
        dsd_generated,
        dsd_throughput,
        dsd_time,
        dsd_acceptance_rate,
        dsd_total_comm,
        dsd_total_dup_bytes,
        dsd_rounds,
        dsd_total_slm,
        dsd_total_llm,
        dsd_total_comm_raw,
        dsd_total_comm_comp,
    ) = generate_DSD(uav_node_dsd, bs_node_dsd, input_ids, tokenizer, dsd_args)

    torch.cuda.empty_cache()
    _prefix, t_ar_llm, tp_ar_llm = normal_generate(target_model, tokenizer, input_ids, device_2, args)

    torch.cuda.empty_cache()
    _prefix, t_ar_slm, tp_ar_slm = normal_generate(draft_model, tokenizer, input_ids, device_1, args)

    # 计算 eta 表字段（分别给 Vanilla DSSD / Optimized DSSD）
    vanilla_T_draft_ms = (vanilla_dssd_total_slm / vanilla_dssd_rounds) * 1000 if vanilla_dssd_rounds > 0 else 0.0
    vanilla_T_comm_ms = (vanilla_dssd_total_comm / vanilla_dssd_rounds) * 1000 if vanilla_dssd_rounds > 0 else 0.0
    vanilla_T_verify_ms = (vanilla_dssd_total_llm / vanilla_dssd_rounds) * 1000 if vanilla_dssd_rounds > 0 else 0.0

    opt_T_draft_ms = (opt_dssd_total_slm / opt_dssd_rounds) * 1000 if opt_dssd_rounds > 0 else 0.0
    opt_T_comm_ms = (opt_dssd_total_comm / opt_dssd_rounds) * 1000 if opt_dssd_rounds > 0 else 0.0
    opt_T_verify_ms = (opt_dssd_total_llm / opt_dssd_rounds) * 1000 if opt_dssd_rounds > 0 else 0.0

    dynamic_T_draft_ms = (dynamic_dssd_total_slm / dynamic_dssd_rounds) * 1000 if dynamic_dssd_rounds > 0 else 0.0
    dynamic_T_comm_ms = (dynamic_dssd_total_comm / dynamic_dssd_rounds) * 1000 if dynamic_dssd_rounds > 0 else 0.0
    dynamic_T_verify_ms = (dynamic_dssd_total_llm / dynamic_dssd_rounds) * 1000 if dynamic_dssd_rounds > 0 else 0.0

    async_dynamic_T_draft_ms = (async_dynamic_dssd_total_slm / async_dynamic_dssd_rounds) * 1000 if async_dynamic_dssd_rounds > 0 else 0.0
    async_dynamic_T_comm_ms = (async_dynamic_dssd_total_comm / async_dynamic_dssd_rounds) * 1000 if async_dynamic_dssd_rounds > 0 else 0.0
    async_dynamic_T_verify_ms = (async_dynamic_dssd_total_llm / async_dynamic_dssd_rounds) * 1000 if async_dynamic_dssd_rounds > 0 else 0.0

    vanilla_T_total_ms = vanilla_T_draft_ms + vanilla_T_comm_ms + vanilla_T_verify_ms
    opt_T_total_ms = opt_T_draft_ms + opt_T_comm_ms + opt_T_verify_ms
    dynamic_T_total_ms = dynamic_T_draft_ms + dynamic_T_comm_ms + dynamic_T_verify_ms
    async_dynamic_T_total_ms = async_dynamic_T_draft_ms + async_dynamic_T_comm_ms + async_dynamic_T_verify_ms

    print("\n=== Summary Comparison ===")
    print(f"Vanilla DSSD       | Throughput: {vanilla_dssd_throughput:.2f} tok/s | AccRate: {vanilla_dssd_acceptance_rate:.3f} | Comm: {vanilla_dssd_total_comm:.3f}s | SLM: {vanilla_dssd_total_slm:.3f}s | LLM: {vanilla_dssd_total_llm:.3f}s")
    print(f"Optimized DSSD     | Throughput: {opt_dssd_throughput:.2f} tok/s | AccRate: {opt_dssd_acceptance_rate:.3f} | Comm: {opt_dssd_total_comm:.3f}s | SLM: {opt_dssd_total_slm:.3f}s | LLM: {opt_dssd_total_llm:.3f}s")
    print(f"Dynamic DSSD       | Throughput: {dynamic_dssd_throughput:.2f} tok/s | AccRate: {dynamic_dssd_acceptance_rate:.3f} | Comm: {dynamic_dssd_total_comm:.3f}s | SLM: {dynamic_dssd_total_slm:.3f}s | LLM: {dynamic_dssd_total_llm:.3f}s")
    print(f"Async Dynamic DSSD | Throughput: {async_dynamic_dssd_throughput:.2f} tok/s | AccRate: {async_dynamic_dssd_acceptance_rate:.3f} | Comm: {async_dynamic_dssd_total_comm:.3f}s | SLM: {async_dynamic_dssd_total_slm:.3f}s | LLM: {async_dynamic_dssd_total_llm:.3f}s")
    print(f"DSD                | Throughput: {dsd_throughput:.2f} tok/s | AccRate: {dsd_acceptance_rate:.3f} | Comm(raw): {dsd_total_comm_raw:.3f}s | Comm(summary): {dsd_total_comm_comp:.3f}s")
    
    recorder.add_entry(
        model_s=args.draft_model_name.rstrip('/').split('/')[-1],
        model_l=args.target_model_name.rstrip('/').split('/')[-1],
        draft_model=args.draft_model_name,
        target_model=args.target_model_name,
        workload=args.workload,
        prompt_id=args.prompt_id,
        input_hash=_short_input_hash(args.input),
        seed=args.seed,
        run_tag=args.run_tag,
        timestamp=datetime.now().isoformat(timespec="seconds"),

        speedup_vanilla_dssd=round(vanilla_dssd_throughput / max(tp_ar_llm, 1e-8), 2),
        speedup_opt_dssd=round(opt_dssd_throughput / max(tp_ar_llm, 1e-8), 2),
        speedup_dynamic_dssd=round(dynamic_dssd_throughput / max(tp_ar_llm, 1e-8), 2) if args.enable_dynamic_dssd else 0.0,
        speedup_async_dynamic_dssd=round(async_dynamic_dssd_throughput / max(tp_ar_llm, 1e-8), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        speedup_dsd=round(dsd_throughput / max(tp_ar_llm, 1e-8), 2),

        b_vanilla_dssd=round(vanilla_dssd_total_comm / max(t_ar_slm, 1e-4), 3),
        b_opt_dssd=round(opt_dssd_total_comm / max(t_ar_slm, 1e-4), 3),
        b_dynamic_dssd=round(dynamic_dssd_total_comm / max(t_ar_slm, 1e-4), 3) if args.enable_dynamic_dssd else 0.0,
        b_async_dynamic_dssd=round(async_dynamic_dssd_total_comm / max(t_ar_slm, 1e-4), 3) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        b_dsd=round(dsd_total_comm / max(t_ar_slm, 1e-4), 3),

        c_vanilla_dssd=round(t_ar_slm / max(t_ar_llm, 1e-8), 3),
        c_opt_dssd=round(t_ar_slm / max(t_ar_llm, 1e-8), 3),
        c_dynamic_dssd=round(t_ar_slm / max(t_ar_llm, 1e-8), 3),
        c_async_dynamic_dssd=round(t_ar_slm / max(t_ar_llm, 1e-8), 3),
        c_dsd=round(t_ar_slm / max(t_ar_llm, 1e-8), 3),

        accept_rate_vanilla_dssd=round(vanilla_dssd_acceptance_rate, 3),
        accept_rate_opt_dssd=round(opt_dssd_acceptance_rate, 3),
        accept_rate_dynamic_dssd=round(dynamic_dssd_acceptance_rate, 3) if args.enable_dynamic_dssd else 0.0,
        accept_rate_async_dynamic_dssd=round(async_dynamic_dssd_acceptance_rate, 3) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        accept_rate_dsd=round(dsd_acceptance_rate, 3),

        thr_vanilla_dssd=round(vanilla_dssd_throughput, 4),
        thr_opt_dssd=round(opt_dssd_throughput, 4),
        thr_dynamic_dssd=round(dynamic_dssd_throughput, 4) if args.enable_dynamic_dssd else 0.0,
        thr_async_dynamic_dssd=round(async_dynamic_dssd_throughput, 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        thr_dsd=round(dsd_throughput, 4),
        base_thr=round(tp_ar_llm, 4),
        slm_thr=round(tp_ar_slm, 4),

        gamma=args.gamma,
        dynamic_gamma_set=args.dynamic_gamma_set,
        dynamic_min_gamma=args.dynamic_min_gamma,
        dynamic_beta=args.dynamic_beta,
        dynamic_z_alpha=args.dynamic_z_alpha,
        dynamic_lambda_sw=args.dynamic_lambda_sw,
        dynamic_lambda_cmp=args.dynamic_lambda_cmp,
        dynamic_lambda_tail=args.dynamic_lambda_tail,
        dynamic_hysteresis=args.dynamic_hysteresis,
        dynamic_min_residency=args.dynamic_min_residency,
        dynamic_alpha_panic=args.dynamic_alpha_panic,
        dynamic_safe_gamma=args.dynamic_safe_gamma,
        dynamic_warmup_rounds=args.dynamic_warmup_rounds,
        rtt_ms=args.rtt * 1e3,
        bw_Mbps=args.bandwidth,
        link_trace_path=args.link_trace_path,
        link_trace_speedup=args.link_trace_speedup,
        prompt_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        reject_topk=args.reject_topk,
        reject_mass_threshold=args.reject_mass_threshold,
        reject_packet_dtype=args.reject_packet_dtype,
        reject_min_prob=args.reject_min_prob,
        compress_reject_packet=args.compress_reject_packet,
        async_shadow_enabled=args.async_shadow_enabled,
        async_shadow_gamma_cap=args.async_shadow_gamma_cap,

        t_ar_slm=round(t_ar_slm, 4),
        t_ar_llm=round(t_ar_llm, 4),

        dup_B_vanilla_dssd=round(vanilla_dssd_total_dup_bytes / 1024, 1),
        dup_B_opt_dssd=round(opt_dssd_total_dup_bytes / 1024, 1),
        dup_B_dynamic_dssd=round(dynamic_dssd_total_dup_bytes / 1024, 1) if args.enable_dynamic_dssd else 0.0,
        dup_B_async_dynamic_dssd=round(async_dynamic_dssd_total_dup_bytes / 1024, 1) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        dup_B_dsd=round(dsd_total_dup_bytes / 1024, 1),

        rounds_vanilla_dssd=vanilla_dssd_rounds,
        rounds_opt_dssd=opt_dssd_rounds,
        rounds_dynamic_dssd=dynamic_dssd_rounds if args.enable_dynamic_dssd else 0,
        rounds_async_dynamic_dssd=async_dynamic_dssd_rounds if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        rounds_dsd=dsd_rounds,

        time_vanilla_dssd=round(vanilla_dssd_time, 4),
        time_opt_dssd=round(opt_dssd_time, 4),
        time_dynamic_dssd=round(dynamic_dssd_time, 4) if args.enable_dynamic_dssd else 0.0,
        time_async_dynamic_dssd=round(async_dynamic_dssd_time, 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        time_dsd=round(dsd_time, 4),

        T_draft_ms_vanilla=round(vanilla_T_draft_ms, 2),
        T_comm_ms_vanilla=round(vanilla_T_comm_ms, 2),
        T_verify_ms_vanilla=round(vanilla_T_verify_ms, 2),
        T_total_ms_vanilla=round(vanilla_T_total_ms, 2),

        T_draft_ms_opt=round(opt_T_draft_ms, 2),
        T_comm_ms_opt=round(opt_T_comm_ms, 2),
        T_verify_ms_opt=round(opt_T_verify_ms, 2),
        T_total_ms_opt=round(opt_T_total_ms, 2),

        T_draft_ms_dynamic=round(dynamic_T_draft_ms, 2) if args.enable_dynamic_dssd else 0.0,
        T_comm_ms_dynamic=round(dynamic_T_comm_ms, 2) if args.enable_dynamic_dssd else 0.0,
        T_verify_ms_dynamic=round(dynamic_T_verify_ms, 2) if args.enable_dynamic_dssd else 0.0,
        T_total_ms_dynamic=round(dynamic_T_total_ms, 2) if args.enable_dynamic_dssd else 0.0,

        T_draft_ms_async_dynamic=round(async_dynamic_T_draft_ms, 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        T_comm_ms_async_dynamic=round(async_dynamic_T_comm_ms, 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        T_verify_ms_async_dynamic=round(async_dynamic_T_verify_ms, 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        T_total_ms_async_dynamic=round(async_dynamic_T_total_ms, 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,

        round_total_mean_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_total_latency_mean_ms", 0.0), 2),
        round_total_p50_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_total_latency_p50_ms", 0.0), 2),
        round_total_p95_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_total_latency_p95_ms", 0.0), 2),
        round_total_p99_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_total_latency_p99_ms", 0.0), 2),
        round_comm_mean_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_comm_latency_mean_ms", 0.0), 2),
        round_slm_mean_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_slm_latency_mean_ms", 0.0), 2),
        round_llm_mean_ms_vanilla=round(vanilla_dssd_reject_stats.get("round_llm_latency_mean_ms", 0.0), 2),

        round_total_mean_ms_opt=round(opt_dssd_reject_stats.get("round_total_latency_mean_ms", 0.0), 2),
        round_total_p50_ms_opt=round(opt_dssd_reject_stats.get("round_total_latency_p50_ms", 0.0), 2),
        round_total_p95_ms_opt=round(opt_dssd_reject_stats.get("round_total_latency_p95_ms", 0.0), 2),
        round_total_p99_ms_opt=round(opt_dssd_reject_stats.get("round_total_latency_p99_ms", 0.0), 2),
        round_comm_mean_ms_opt=round(opt_dssd_reject_stats.get("round_comm_latency_mean_ms", 0.0), 2),
        round_slm_mean_ms_opt=round(opt_dssd_reject_stats.get("round_slm_latency_mean_ms", 0.0), 2),
        round_llm_mean_ms_opt=round(opt_dssd_reject_stats.get("round_llm_latency_mean_ms", 0.0), 2),

        round_total_mean_ms_dynamic=round(dynamic_dssd_stats.get("round_total_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_total_p50_ms_dynamic=round(dynamic_dssd_stats.get("round_total_latency_p50_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_total_p95_ms_dynamic=round(dynamic_dssd_stats.get("round_total_latency_p95_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_total_p99_ms_dynamic=round(dynamic_dssd_stats.get("round_total_latency_p99_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_comm_mean_ms_dynamic=round(dynamic_dssd_stats.get("round_comm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_slm_mean_ms_dynamic=round(dynamic_dssd_stats.get("round_slm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,
        round_llm_mean_ms_dynamic=round(dynamic_dssd_stats.get("round_llm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd else 0.0,

        round_total_mean_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_total_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_total_p50_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_total_latency_p50_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_total_p95_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_total_latency_p95_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_total_p99_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_total_latency_p99_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_comm_mean_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_comm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_slm_mean_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_slm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        round_llm_mean_ms_async_dynamic=round(async_dynamic_dssd_stats.get("round_llm_latency_mean_ms", 0.0), 2) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,

        dsd_comm_raw_s=round(dsd_total_comm_raw, 4),
        dsd_comm_comp_s=round(dsd_total_comm_comp, 4),

        vanilla_accept_all_rounds=vanilla_dssd_reject_stats.get("accept_all_rounds", 0),
        vanilla_reject_rounds=vanilla_dssd_reject_stats.get("reject_rounds", 0),
        vanilla_reject_full_cnt=vanilla_dssd_reject_stats.get("reject_full_cnt", 0),
        vanilla_reject_down_bytes_actual=vanilla_dssd_reject_stats.get("reject_down_bytes_actual", 0),
        vanilla_reject_down_bytes_full_equiv=vanilla_dssd_reject_stats.get("reject_down_bytes_full_equiv", 0),
        vanilla_avg_kept_mass=round(vanilla_dssd_reject_stats.get("avg_kept_mass", 0.0), 6),

        opt_accept_all_rounds=opt_dssd_reject_stats.get("accept_all_rounds", 0),
        opt_reject_rounds=opt_dssd_reject_stats.get("reject_rounds", 0),
        opt_reject_compressed_cnt=opt_dssd_reject_stats.get("reject_compressed_cnt", 0),
        opt_reject_full_cnt=opt_dssd_reject_stats.get("reject_full_cnt", 0),
        opt_compression_hit_rate=round(opt_dssd_reject_stats.get("compression_hit_rate", 0.0), 4),
        opt_reject_down_bytes_actual=opt_dssd_reject_stats.get("reject_down_bytes_actual", 0),
        opt_reject_down_bytes_full_equiv=opt_dssd_reject_stats.get("reject_down_bytes_full_equiv", 0),
        opt_reject_downlink_saving=round(opt_dssd_reject_stats.get("reject_downlink_saving", 0.0), 4),
        opt_avg_kept_mass=round(opt_dssd_reject_stats.get("avg_kept_mass", 0.0), 6),
        opt_avg_tail_mass=round(opt_dssd_reject_stats.get("avg_tail_mass", 0.0), 6),

        dynamic_accept_all_rounds=dynamic_dssd_stats.get("accept_all_rounds", 0) if args.enable_dynamic_dssd else 0,
        dynamic_reject_rounds=dynamic_dssd_stats.get("reject_rounds", 0) if args.enable_dynamic_dssd else 0,
        dynamic_reject_compressed_cnt=dynamic_dssd_stats.get("reject_compressed_cnt", 0) if args.enable_dynamic_dssd else 0,
        dynamic_reject_full_cnt=dynamic_dssd_stats.get("reject_full_cnt", 0) if args.enable_dynamic_dssd else 0,
        dynamic_compression_hit_rate=round(dynamic_dssd_stats.get("compression_hit_rate", 0.0), 4) if args.enable_dynamic_dssd else 0.0,
        dynamic_reject_down_bytes_actual=dynamic_dssd_stats.get("reject_down_bytes_actual", 0) if args.enable_dynamic_dssd else 0,
        dynamic_reject_down_bytes_full_equiv=dynamic_dssd_stats.get("reject_down_bytes_full_equiv", 0) if args.enable_dynamic_dssd else 0,
        dynamic_reject_downlink_saving=round(dynamic_dssd_stats.get("reject_downlink_saving", 0.0), 4) if args.enable_dynamic_dssd else 0.0,
        dynamic_avg_kept_mass=round(dynamic_dssd_stats.get("avg_kept_mass", 0.0), 6) if args.enable_dynamic_dssd else 0.0,
        dynamic_avg_tail_mass=round(dynamic_dssd_stats.get("avg_tail_mass", 0.0), 6) if args.enable_dynamic_dssd else 0.0,
        dynamic_avg_gamma=round(dynamic_dssd_stats.get("avg_gamma", 0.0), 4) if args.enable_dynamic_dssd else 0.0,
        dynamic_gamma_usage=_json_dumps(dynamic_dssd_stats.get("gamma_usage", {})) if args.enable_dynamic_dssd else "{}",
        dynamic_compression_on_ratio=round(dynamic_dssd_stats.get("compression_on_ratio", 0.0), 4) if args.enable_dynamic_dssd else 0.0,
        dynamic_controller_switch_count=dynamic_dssd_stats.get("controller_switch_count", 0) if args.enable_dynamic_dssd else 0,
        dynamic_controller_switch_rate=round(dynamic_dssd_stats.get("controller_switch_rate", 0.0), 4) if args.enable_dynamic_dssd else 0.0,
        dynamic_controller_safe_mode_count=dynamic_dssd_stats.get("controller_safe_mode_count", 0) if args.enable_dynamic_dssd else 0,
        dynamic_controller_safe_mode_rate=round(dynamic_dssd_stats.get("controller_safe_mode_rate", 0.0), 4) if args.enable_dynamic_dssd else 0.0,

        async_dynamic_avg_gamma=round(async_dynamic_dssd_stats.get("avg_gamma", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_dynamic_gamma_usage=_json_dumps(async_dynamic_dssd_stats.get("gamma_usage", {})) if args.enable_dynamic_dssd and args.async_shadow_enabled else "{}",
        async_dynamic_compression_on_ratio=round(async_dynamic_dssd_stats.get("compression_on_ratio", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_dynamic_controller_switch_count=async_dynamic_dssd_stats.get("controller_switch_count", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_dynamic_controller_switch_rate=round(async_dynamic_dssd_stats.get("controller_switch_rate", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_dynamic_controller_safe_mode_count=async_dynamic_dssd_stats.get("controller_safe_mode_count", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_dynamic_controller_safe_mode_rate=round(async_dynamic_dssd_stats.get("controller_safe_mode_rate", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,

        async_shadow_launches=async_dynamic_dssd_stats.get("async_shadow_launches", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_launch_rate=round(async_dynamic_dssd_stats.get("async_shadow_launch_rate", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_hits=async_dynamic_dssd_stats.get("async_shadow_hits", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_hit_rate=round(async_dynamic_dssd_stats.get("async_shadow_hit_rate", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_misses=async_dynamic_dssd_stats.get("async_shadow_misses", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_discard_reject=async_dynamic_dssd_stats.get("async_shadow_discard_reject", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_discard_mismatch=async_dynamic_dssd_stats.get("async_shadow_discard_mismatch", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_discard_gamma_change=async_dynamic_dssd_stats.get("async_shadow_discard_gamma_change", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_discard_prefix_mismatch=async_dynamic_dssd_stats.get("async_shadow_discard_prefix_mismatch", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_discard_generation_end=async_dynamic_dssd_stats.get("async_shadow_discard_generation_end", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_partial_extensions=async_dynamic_dssd_stats.get("async_shadow_partial_extensions", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_eligible_rounds=async_dynamic_dssd_stats.get("async_shadow_eligible_rounds", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_blocked_overlap=async_dynamic_dssd_stats.get("async_shadow_blocked_overlap", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_blocked_benefit=async_dynamic_dssd_stats.get("async_shadow_blocked_benefit", 0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0,
        async_shadow_total_time_s=round(async_dynamic_dssd_stats.get("async_shadow_total_time_s", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_wait_time_s=round(async_dynamic_dssd_stats.get("async_shadow_wait_time_s", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_wasted_compute_time_s=round(async_dynamic_dssd_stats.get("async_shadow_wasted_compute_time_s", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_reused_compute_time_s=round(async_dynamic_dssd_stats.get("async_shadow_reused_compute_time_s", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_saved_draft_time_s=round(async_dynamic_dssd_stats.get("async_shadow_saved_draft_time_s", 0.0), 4) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_anchor_mode=async_dynamic_dssd_stats.get("async_shadow_anchor_mode", "sample") if args.enable_dynamic_dssd and args.async_shadow_enabled else "",
        async_shadow_overlap_margin=async_dynamic_dssd_stats.get("async_shadow_overlap_margin", 1.0) if args.enable_dynamic_dssd and args.async_shadow_enabled else 0.0,
        async_shadow_reused_gamma_usage=_json_dumps(async_dynamic_dssd_stats.get("async_shadow_reused_gamma_usage", {})) if args.enable_dynamic_dssd and args.async_shadow_enabled else "{}",
    )
