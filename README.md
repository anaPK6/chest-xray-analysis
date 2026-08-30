# AI-Powered Chest X-ray Analysis & Radiology Report Generation

A chest X-ray reader that classifies thoracic disease, shows *where* it looked,
writes the radiology report, and answers follow-up questions about the scan.

Upload an X-ray → validate it's actually a chest X-ray → multi-label
classification → per-class Grad-CAM → structured findings → LLM radiology
report → RAG chatbot → PDF export.

> **Research/education demo only. Not a diagnostic tool.**
> No image ever leaves the machine — the classifier and the LLM both run locally.

## Why

A raw disease probability isn't enough to trust a medical model. You need to
know what it saw, where it looked, have it explained in the language of a
radiology report, and be able to ask follow-up questions. This is one pipeline
that does all four.

## Results

DenseNet121 trained from scratch on CheXpert — **mean AUROC 0.847** across five
pathologies (10,000-image subset, 2 epochs, trained locally on an M4 Mac).

| Pathology | AUROC |
|---|---|
| Edema | 0.897 |
| Consolidation | 0.889 |
| Pleural Effusion | 0.887 |
| Atelectasis | 0.794 |
| Cardiomegaly | 0.769 |
| **Mean** | **0.847** |

Evaluated on the 234-image CheXpert validation split. AUROC is the headline
metric because it's threshold-independent — with class imbalance this strong,
sensitivity at a fixed 0.5 threshold is misleading, and the in-app evaluation
view says so explicitly rather than hiding it.

## Features

**Multi-label classification** — DenseNet121 over 5 thoracic pathologies.
CheXpert's uncertainty labels (`-1`) are mapped per-label with a U-Ones/U-Zeros
policy recorded in `model_config.json`.

**Per-class Grad-CAM** — click any pathology to fetch that class's heatmap on
demand, with an opacity slider over the scan. Explanations are per-prediction,
not one generic saliency map.

**Local LLM radiology report** — confidence-thresholded findings go to Llama 3.1
via Ollama, which writes a structured report. Editable in-app, exportable to PDF.

**RAG chatbot** — "Ask about this X-ray." Scan-specific questions are grounded in
the actual findings and probabilities; general medical questions retrieve from a
curated knowledge base (5 pathologies × 4 sections) via cosine similarity over
`nomic-embed-text` embeddings, with source citations. No vector DB — the corpus
is small enough that an in-memory index is the honest choice.

**Non-X-ray input guard** — public uploads include selfies, memes, and wrong body
parts, on which the model would happily emit confident nonsense. A pixel-level
heuristic (hard grayscale gate + intensity distribution + aspect ratio) rejects
them with a `422` *before* the model ever runs. Real X-rays are true monochrome;
photos carry per-pixel color variance even when they look neutral.

**Evaluation view** — ROC curves, per-label AUROC, and confusion metrics rendered
from a precomputed `evaluation.json`.

## Stack

PyTorch · DenseNet121 · Grad-CAM · CheXpert · FastAPI · React (Vite) · Ollama

## Setup

```bash
# backend
conda create -n medical-ai python=3.11 -y
conda activate medical-ai
pip install -r requirements.txt

# LLM backend — report + chat both need this running
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve                      # port 11434, does not auto-start

# run the API (port 8010)
MODEL_SOURCE=local PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uvicorn backend.main:app --port 8010

# frontend, separate terminal
cd frontend && npm install && npm run dev     # http://localhost:5173
```

The Vite dev server proxies `/api/*` → FastAPI on `:8010`.

The trained checkpoint is committed, so this runs as-is — no training required.

## Model sources

Switch with `MODEL_SOURCE`:

- **`local`** (default for this repo) — the included DenseNet121 from
  `models/checkpoints/`, ImageNet normalization, 3-channel.
- **`pretrained`** — torchxrayvision DenseNet, xrv normalization, 1-channel.

Both sit behind one `.labels` / `.predict()` / `.norm` interface.
Normalization is the subtle failure mode here: the two sources expect different
input scaling and channel counts, and a mismatch produces confident garbage
rather than an error. `utils/preprocess.py` reads the norm off the model itself,
so the two can't silently diverge.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /model` | active source, architecture, per-label AUROC |
| `GET /evaluation` | precomputed ROC / confusion metrics |
| `POST /analyze` | guard check → probabilities → findings |
| `POST /gradcam` | heatmap for one pathology |
| `POST /report` | LLM radiology report |
| `POST /chat` | RAG chat turn |
| `POST /pdf` | export report |

## Training your own

CheXpert isn't included (~22GB). Download `CheXpert-v1.0-small` into `data/`,
then:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_local.py --subset 10000 --epochs 2
python evaluate_local.py          # refresh evaluation.json
```

`train_local.py` writes the handoff contract (`state_dict.pth`, `labels.json`,
`model_config.json`) to `models/checkpoints/`. Roughly 10 minutes on an M4.
For reference, published work reaches ~0.85–0.90 training on the full 223k set.

`notebooks/chexpert_train_kaggle.ipynb` is the cloud-GPU equivalent, kept for
anyone with a Kaggle setup.

## Layout

```
backend/           FastAPI app
frontend/          Vite + React SPA
models/            classifier, findings, checkpoints/
explainability/    Grad-CAM + overlay
report_generator/  LLM interface, knowledge base, retriever, chat
utils/             config, preprocess, pipeline, X-ray guard, PDF export
train_local.py     training
evaluate_local.py  evaluation metrics
```

## Notes

Grad-CAM hooks `model.features.denseblock4`, not `model.features` —
torchxrayvision applies an in-place ReLU after the features container, which
breaks a backward hook on a view. The saved activations get their own ReLU in
the CAM to compensate.

The report generator is behind an ABC (`report_generator/interface.py`) so
another LLM backend can be added, though Ollama is the only one wired up.
