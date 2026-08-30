"""FastAPI backend wrapping the chest X-ray pipeline.

Endpoints:
  GET  /health            -> liveness + device + available LLM providers
  GET  /model             -> model info (source, arch, mean/per-label AUROC)
  POST /analyze           -> probabilities, findings, top-finding Grad-CAM (base64 PNG)
  POST /gradcam           -> Grad-CAM (base64 PNG) for one chosen pathology
  POST /report            -> LLM radiology report for given findings + provider
  POST /chat              -> RAG chatbot: answer a question about the X-ray + sources
  POST /pdf               -> PDF bytes for a report

The heavy vision model loads once at startup; the LLM provider is chosen per
request so the UI toggle can switch Ollama/OpenAI live.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

from utils.config import DEVICE
from utils.pipeline import Pipeline
from utils.pdf_export import report_to_pdf
from utils.xray_guard import xray_likeness
from report_generator import AVAILABLE_PROVIDERS

app = FastAPI(title="AI Chest X-ray Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    # backend runs on :8010 (port 8000 is used by another local app)
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None


def pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()  # loads the vision model once
    return _pipeline


def _png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "providers": AVAILABLE_PROVIDERS,
        "default_provider": "ollama",
    }


@app.get("/model")
def model():
    return pipeline().clf.model_info()


@app.get("/evaluation")
def evaluation():
    """Serve the saved validation-set evaluation (ROC, AUROC, confusion, info)."""
    from utils.config import CHECKPOINT_DIR
    path = CHECKPOINT_DIR / "evaluation.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation.json — run `python evaluate_local.py` first.")
    import json
    return json.loads(path.read_text())


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read image")

    # Reject non-chest-X-ray uploads up front — don't run the model on them.
    guard = xray_likeness(pil_img)
    if not guard["is_xray"]:
        reason = guard["reasons"][0] if guard["reasons"] else "does not look like a chest X-ray"
        raise HTTPException(422, {
            "error": "not_a_chest_xray",
            "message": f"This image was rejected — {reason}. "
                       "Please upload a frontal chest X-ray.",
            "guard": guard,
        })

    result = pipeline().analyze(pil_img)
    return {
        "probs": result["probs"],
        "findings": result["findings"],
        "overlay": _png_b64(result["overlay"]) if result["overlay"] else None,
        "overlay_label": result["overlay_label"],
        "guard": guard,
    }


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), label: str = Form(...)):
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read image")
    overlay = pipeline().gradcam(pil_img, label)
    if overlay is None:
        raise HTTPException(400, f"unknown pathology: {label}")
    return {"label": label, "overlay": _png_b64(overlay)}


class ReportRequest(BaseModel):
    findings: dict
    provider: str | None = None


@app.post("/report")
def report(req: ReportRequest):
    if req.provider and req.provider not in AVAILABLE_PROVIDERS:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    try:
        text = pipeline().report(req.findings, req.provider)
    except Exception as e:  # e.g. Ollama not running / no API key
        raise HTTPException(502, f"report generation failed: {e}")
    return {"report": text, "provider": req.provider or "ollama"}


class ChatRequest(BaseModel):
    message: str
    findings: dict = {}
    history: list[dict] = []


@app.post("/chat")
def chat(req: ChatRequest):
    from report_generator.chat import answer
    try:
        return answer(req.message, req.findings, req.history)
    except Exception as e:  # Ollama down / embed failure
        raise HTTPException(502, f"chat failed: {e}")


class PdfRequest(BaseModel):
    report: str


@app.post("/pdf")
def pdf(req: PdfRequest):
    data = report_to_pdf(req.report)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=chest_xray_report.pdf"},
    )
