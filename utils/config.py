"""Central configuration for the medical-ai app."""
import os
import torch

# ---- Device -----------------------------------------------------------------
# Apple Silicon (M-series) -> MPS. Falls back to CPU if unavailable.
# PYTORCH_ENABLE_MPS_FALLBACK lets unsupported ops run on CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# ---- Model ------------------------------------------------------------------
# MODEL_SOURCE selects where the classifier weights come from:
#   "pretrained" -> torchxrayvision DenseNet (v1 default, no training needed)
#   "local"      -> your Kaggle-trained model in models/checkpoints/ (handoff
#                   contract: state_dict.pth + model_config.json + labels.json)
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "pretrained")  # "pretrained" | "local"

# pretrained: torchxrayvision weights id ("all" = CheXpert/NIH/MIMIC etc.)
PRETRAINED_WEIGHTS = "densenet121-res224-all"

# local: directory holding the three handoff-contract artifacts
import pathlib
CHECKPOINT_DIR = pathlib.Path(__file__).resolve().parent.parent / "models" / "checkpoints"

INPUT_SIZE = 224

# ---- Findings ---------------------------------------------------------------
# Probability at/above this -> reported as a positive finding.
CONFIDENCE_THRESHOLD = 0.5

# ---- LLM (Ollama — local, offline) ------------------------------------------
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"
