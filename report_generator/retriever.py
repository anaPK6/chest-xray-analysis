"""In-memory RAG retriever over the curated pathology KB.

Embeds the ~20 KB chunks once (lazily) via Ollama's nomic-embed-text, then
answers queries by cosine similarity. For a corpus this small an in-memory
NumPy index is simpler and faster than standing up a vector DB.
"""
from __future__ import annotations

import numpy as np
import requests

from utils.config import OLLAMA_HOST
from report_generator.knowledge_base import kb_chunks

EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    v = np.asarray(r.json()["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v  # normalize -> cosine == dot product


class Retriever:
    def __init__(self):
        self.chunks = kb_chunks()
        self._matrix: np.ndarray | None = None  # built lazily (needs Ollama up)

    def _ensure_index(self):
        if self._matrix is None:
            self._matrix = np.vstack([_embed(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k KB chunks for a query, with similarity scores."""
        self._ensure_index()
        q = _embed(query)
        sims = self._matrix @ q  # cosine, both normalized
        top = np.argsort(-sims)[:k]
        out = []
        for i in top:
            c = self.chunks[int(i)]
            out.append({**c, "score": float(sims[int(i)])})
        return out
