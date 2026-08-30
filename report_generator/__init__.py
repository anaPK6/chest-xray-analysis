"""Report generator factory. Ollama-only (local, offline).

The strategy-pattern interface is kept so another backend could be re-added,
but OpenAI has been removed — this app runs fully local via Ollama.
"""
from __future__ import annotations

from report_generator.interface import ReportGenerator

AVAILABLE_PROVIDERS = ["ollama"]


def get_generator(provider: str | None = None) -> ReportGenerator:
    provider = provider or "ollama"
    if provider == "ollama":
        from report_generator.ollama import OllamaGenerator
        return OllamaGenerator()
    raise ValueError(f"unknown provider: {provider!r}")
