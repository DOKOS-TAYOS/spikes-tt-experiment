from __future__ import annotations

import pytest

from spike_mps.labels import compute_label


@pytest.mark.parametrize(
    ("spike_train", "expected"),
    [
        ("00000", 0),
        ("10101", 3),
        ("11111", 5),
    ],
)
def test_count_ones_returns_number_of_ones(
    spike_train: str,
    expected: int,
) -> None:
    assert compute_label("count_ones", spike_train) == expected


@pytest.mark.parametrize(
    ("spike_train", "expected"),
    [
        ("00000", 5),
        ("10101", 2),
        ("11111", 0),
    ],
)
def test_count_zeros_returns_number_of_zeros(
    spike_train: str,
    expected: int,
) -> None:
    assert compute_label("count_zeros", spike_train) == expected


@pytest.mark.parametrize(
    ("spike_train", "expected"),
    [
        ("1101", 1),
        ("11101", 2),
        ("111011", 3),
        ("10101", 0),
        ("11111", 4),
    ],
)
def test_adjacent_ones_score_counts_adjacent_pairs(
    spike_train: str,
    expected: int,
) -> None:
    assert compute_label("adjacent_ones_score", spike_train) == expected


def test_compute_label_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        compute_label("not_a_task", "10101")
