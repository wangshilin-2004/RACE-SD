"""Dependency-free PipeSD trigger and token-batch scheduling helpers."""

from typing import List


def pipesd_should_verify(
    confidences: List[float], single_threshold: float, multi_threshold: float
) -> bool:
    if not confidences:
        return False
    product = 1.0
    for value in confidences:
        product *= float(value)
    return (
        float(confidences[-1]) < float(single_threshold)
        or product < float(multi_threshold)
    )


def pipesd_dynamic_token_schedule(
    token_compute_times: List[float], fixed_cost_s: float, per_token_s: float
):
    count = len(token_compute_times)
    if count == 0:
        return [], 0.0
    ready = []
    elapsed = 0.0
    for value in token_compute_times:
        elapsed += max(0.0, float(value))
        ready.append(elapsed)

    completion = [0.0] * count
    predecessor = [0] * count
    for end in range(count):
        best_time = float("inf")
        best_start = 0
        for start in range(end + 1):
            previous = completion[start - 1] if start else 0.0
            begin = max(previous, ready[end])
            finish = begin + float(fixed_cost_s) + (end - start + 1) * float(per_token_s)
            if finish < best_time:
                best_time = finish
                best_start = start
        completion[end] = best_time
        predecessor[end] = best_start

    batches = []
    end = count - 1
    while end >= 0:
        start = predecessor[end]
        batches.append(list(range(start, end + 1)))
        end = start - 1
    batches.reverse()
    return batches, completion[-1]
