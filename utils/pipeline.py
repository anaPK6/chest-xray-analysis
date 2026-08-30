"""End-to-end pipeline: image -> probs -> findings -> Grad-CAM -> report.

The Streamlit app and CLI both call run_pipeline(); nothing downstream knows
which model or which LLM produced the result.
"""
from __future__ import annotations

from PIL import Image

from models.classifier import XRayClassifier
from models.findings import to_findings
from explainability.gradcam import GradCAM
from explainability.overlay import overlay_heatmap
from utils.preprocess import preprocess
from report_generator import get_generator


class Pipeline:
    """Model + Grad-CAM are loaded once; the LLM backend is chosen per request
    so the UI provider toggle can switch Ollama/OpenAI live."""

    def __init__(self):
        self.clf = XRayClassifier()
        self.cam = GradCAM(self.clf.model)

    def analyze(self, pil_img: Image.Image) -> dict:
        """Vision half: probs + findings + Grad-CAM for the top finding.

        Per-pathology heatmaps are fetched on demand via gradcam() so this
        initial call stays fast.
        """
        img = preprocess(pil_img, norm=self.clf.norm)
        probs = self.clf.predict(img)
        findings = to_findings(probs)

        overlay, overlay_label = None, None
        if findings["top"] is not None:
            overlay_label = findings["top"][0]
            overlay = self._cam_overlay(pil_img, img, overlay_label)

        return {
            "probs": probs,
            "findings": findings,
            "overlay": overlay,
            "overlay_label": overlay_label,
        }

    def gradcam(self, pil_img: Image.Image, label: str) -> Image.Image | None:
        """On-demand Grad-CAM overlay (PIL image) for one pathology."""
        if label not in self.clf.labels:
            return None
        img = preprocess(pil_img, norm=self.clf.norm)
        return self._cam_overlay(pil_img, img, label)

    def _cam_overlay(self, pil_img: Image.Image, arr, label: str):
        class_idx = self.clf.labels.index(label)
        heatmap = self.cam(arr, class_idx)
        return overlay_heatmap(pil_img, heatmap)

    def report(self, findings: dict, provider: str | None = None) -> str:
        """LLM half: findings -> report using the chosen provider."""
        return get_generator(provider).generate(findings)

    def run(self, pil_img: Image.Image, provider: str | None = None) -> dict:
        result = self.analyze(pil_img)
        result["report"] = self.report(result["findings"], provider)
        return result
