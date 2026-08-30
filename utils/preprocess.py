"""Image preprocessing (Phase 1 input pipeline).

CRITICAL: preprocessing MUST match the model that will consume it (handoff
contract). The two model sources normalize differently:

  pretrained (torchxrayvision) -> 1-channel, xrv.normalize -> [-1024, 1024]
  local (Kaggle DenseNet121)   -> 3-channel, ImageNet mean/std (matches training)

`preprocess(img, norm=...)` picks the right transform. The classifier passes the
norm recorded in model_config.json so the app can't silently mismatch.
"""
from __future__ import annotations

import numpy as np
import torchxrayvision as xrv
from PIL import Image
from skimage.transform import resize

from utils.config import INPUT_SIZE, MODEL_SOURCE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# default norm follows the configured source; classifier can override per-model
_DEFAULT_NORM = "xrv" if MODEL_SOURCE == "pretrained" else "imagenet"


def preprocess(pil_img: Image.Image, norm: str = _DEFAULT_NORM) -> np.ndarray:
    """PIL image -> array ready for the classifier.

    norm="xrv":      returns (1, H, W)  in [-1024, 1024]   (torchxrayvision)
    norm="imagenet": returns (3, H, W)  ImageNet-normalized (torchvision)
    """
    if norm == "xrv":
        img = np.array(pil_img.convert("L"), dtype=np.float32)   # grayscale
        img = xrv.datasets.normalize(img, 255)                   # -> [-1024,1024]
        img = img[None, ...]                                      # (1,H,W)
        img = resize(img, (1, INPUT_SIZE, INPUT_SIZE), preserve_range=True)
        return img.astype(np.float32)

    if norm == "imagenet":
        img = pil_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0          # (H,W,3) in [0,1]
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)   # (3,H,W)

    raise ValueError(f"unknown norm: {norm!r}")
