"""RAG chatbot: answer questions about the uploaded X-ray.

Each turn the LLM is given:
  - the current X-ray's structured findings + probabilities (so it can answer
    about THIS scan), and
  - RAG-retrieved chunks from the curated pathology KB (so general medical
    questions are grounded, not hallucinated).

Returns the answer plus which KB sources were used (for citation in the UI).
"""
from __future__ import annotations

import requests

from utils.config import OLLAMA_HOST, OLLAMA_MODEL
from report_generator.retriever import Retriever

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _findings_summary(findings: dict) -> str:
    if not findings:
        return "No X-ray has been analyzed yet."
    pos = findings.get("positive", {})
    neg = findings.get("negative", {})
    thr = findings.get("threshold", 0.5)
    pos_s = ", ".join(f"{k} ({v:.0%})" for k, v in pos.items()) or "none above threshold"
    neg_s = ", ".join(f"{k} ({v:.0%})" for k, v in list(neg.items())[:5])
    return (
        f"Positive findings (>= {thr:.0%}): {pos_s}. "
        f"Below-threshold: {neg_s}."
    )


def build_chat_prompt(message: str, findings: dict, history: list[dict],
                      sources: list[dict]) -> str:
    context = "\n".join(f"- {s['pathology']} ({s['section']}): "
                        f"{s['text'].split(': ', 1)[-1]}" for s in sources)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return f"""You are a helpful assistant explaining a chest X-ray AI analysis to a
curious user. This is an educational demo, NOT medical advice.

CURRENT X-RAY ANALYSIS:
{_findings_summary(findings)}

REFERENCE KNOWLEDGE (retrieved; use it to ground general medical explanations):
{context or "(none retrieved)"}

CONVERSATION SO FAR:
{convo or "(start of conversation)"}

Guidelines:
- If the question is about THIS X-ray, use the analysis above (probabilities, findings).
- If it's a general medical question, use the reference knowledge; don't invent facts.
- Be concise and clear. Remind the user this is not a diagnosis if they ask for one.

User question: {message}
Answer:"""


def answer(message: str, findings: dict, history: list[dict]) -> dict:
    sources = get_retriever().search(message, k=3)
    prompt = build_chat_prompt(message, findings, history, sources)
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return {
        "answer": r.json()["response"].strip(),
        "sources": [{"pathology": s["pathology"], "section": s["section"],
                     "score": round(s["score"], 3)} for s in sources],
    }
