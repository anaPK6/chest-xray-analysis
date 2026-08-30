"""Shared prompt builder — written ONCE so both LLM backends produce
comparable reports (fair to compare in the write-up)."""
from __future__ import annotations


def build_prompt(findings: dict) -> str:
    pos = findings.get("positive", {})
    neg = findings.get("negative", {})
    thr = findings.get("threshold", 0.5)

    pos_lines = "\n".join(f"  - {k}: {v:.2f}" for k, v in pos.items()) or "  - None above threshold"
    neg_lines = "\n".join(f"  - {k}: {v:.2f}" for k, v in list(neg.items())[:5])

    return f"""You are assisting with drafting a radiology-style report for an
educational demo (NOT for clinical use). Given the model's per-pathology
probabilities from a chest X-ray, write a concise, professional radiology
report with sections: FINDINGS and IMPRESSION.

Positive findings (probability >= {thr}):
{pos_lines}

Other pathologies considered (below threshold, top 5):
{neg_lines}

Guidelines:
- Be concise and use standard radiology phrasing.
- Only assert what the probabilities support; hedge appropriately.
- Do not invent measurements or patient history.
- End with a one-line note that this is an AI-generated educational draft.
"""
