"""Frozen financial-dictionary tone scoring (Loughran-McDonald-style word lists).

This is a small subset frozen into source so the pipeline never depends on an external
download. For research-grade use, replace with the full LM list.
"""
from __future__ import annotations

POSITIVE = {
    "strong", "growth", "record", "momentum", "expansion", "innovation", "robust",
    "improve", "improved", "improving", "outperform", "beat", "exceed", "exceeded",
    "raise", "raised", "guidance", "opportunity", "confident", "deliver", "delivered",
}
NEGATIVE = {
    "weak", "decline", "headwind", "pressure", "challenge", "restructuring", "uncertainty",
    "miss", "missed", "below", "lower", "reduced", "reduction", "loss", "impairment",
    "softness", "deteriorate", "deteriorated", "difficult", "risk", "risks",
}
UNCERTAINTY = {"uncertainty", "uncertain", "may", "might", "approximately", "expect",
               "believe", "could", "pending"}
LITIGATION = {"litigation", "lawsuit", "claim", "settlement", "investigation", "sec"}


def tone_scores(text: str) -> dict[str, float]:
    tokens = str(text).lower().split()
    n = max(1, len(tokens))
    return {
        "n_chars": float(len(str(text))),
        "n_tokens": float(len(tokens)),
        "pos_freq": sum(1 for t in tokens if t in POSITIVE) / n,
        "neg_freq": sum(1 for t in tokens if t in NEGATIVE) / n,
        "uncertainty_freq": sum(1 for t in tokens if t in UNCERTAINTY) / n,
        "litigation_freq": sum(1 for t in tokens if t in LITIGATION) / n,
        "numerical_density": sum(1 for t in tokens if any(c.isdigit() for c in t)) / n,
    }


def tone_net(d: dict[str, float]) -> float:
    return d.get("pos_freq", 0.0) - d.get("neg_freq", 0.0)
