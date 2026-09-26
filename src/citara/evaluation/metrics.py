"""Statistics shared by the two evaluation suites.

Kept in one place so that a latency percentile in the retrieval table and a latency
percentile in the generation table are the same calculation. Two implementations of
"p95" that round differently would make the two artifacts quietly incomparable.
"""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """The value at *fraction* through the sorted sample, in milliseconds.

    Nearest-rank rather than interpolated: with thirty measurements an interpolated p95
    would report a number no request actually took.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return round(ordered[index], 1)


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the values that exist, or None when none of them do.

    A missing score is not a zero. Faithfulness is undefined for an answer that asserted
    nothing, and counting that as zero would punish the system for a different failure than
    the one it had.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
