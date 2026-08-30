"""Ollama backend — free, offline, uses local llama3.1:8b."""
from __future__ import annotations

import requests

from utils.config import OLLAMA_HOST, OLLAMA_MODEL
from report_generator.interface import ReportGenerator
from report_generator.prompts import build_prompt


class OllamaGenerator(ReportGenerator):
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model, self.host = model, host

    def generate(self, findings: dict) -> str:
        prompt = build_prompt(findings)
        r = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
