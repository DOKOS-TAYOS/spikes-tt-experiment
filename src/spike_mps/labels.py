from __future__ import annotations

from collections.abc import Callable

from spike_mps.config import TaskName


def compute_label(task: TaskName | str, spike_train: str) -> int:
    _validate_spike_train(spike_train)

    functions: dict[str, Callable[[str], int]] = {
        "count_ones": count_ones,
        "count_zeros": count_zeros,
        "adjacent_ones_score": adjacent_ones_score,
    }
    try:
        return functions[task](spike_train)
    except KeyError as error:
        raise ValueError(f"Unknown task: {task}") from error


def count_ones(spike_train: str) -> int:
    return spike_train.count("1")


def count_zeros(spike_train: str) -> int:
    return spike_train.count("0")


def adjacent_ones_score(spike_train: str) -> int:
    return sum(
        1
        for current_bit, next_bit in zip(spike_train, spike_train[1:], strict=False)
        if current_bit == "1" and next_bit == "1"
    )


def max_label_for_task(task: TaskName, sequence_length: int) -> int:
    if task in {"count_ones", "count_zeros"}:
        return sequence_length
    return max(sequence_length - 1, 0)


def _validate_spike_train(spike_train: str) -> None:
    if not spike_train:
        raise ValueError("spike_train cannot be empty.")
    if any(bit not in {"0", "1"} for bit in spike_train):
        raise ValueError("spike_train must contain only '0' and '1'.")
