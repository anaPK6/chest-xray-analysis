"""The contract every report backend implements (strategy pattern)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ReportGenerator(ABC):
    @abstractmethod
    def generate(self, findings: dict) -> str:
        """findings dict -> radiology report text."""
        ...
