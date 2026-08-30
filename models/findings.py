"""Structured findings (Phase 3): probabilities -> findings via thresholds."""
from __future__ import annotations

from utils.config import CONFIDENCE_THRESHOLD


def to_findings(probs: dict[str, float], threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Split predictions into positive/negative findings with confidences."""
    positive = {k: v for k, v in probs.items() if v >= threshold}
    negative = {k: v for k, v in probs.items() if v < threshold}
    return {
        "threshold": threshold,
        "positive": dict(sorted(positive.items(), key=lambda kv: -kv[1])),
        "negative": dict(sorted(negative.items(), key=lambda kv: -kv[1])),
        "top": max(probs.items(), key=lambda kv: kv[1]) if probs else None,
    }
