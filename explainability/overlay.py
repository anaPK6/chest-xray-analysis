"""Blend a Grad-CAM heatmap onto the original X-ray for display."""
from __future__ import annotations

import numpy as np
from matplotlib import cm
from PIL import Image

from utils.config import INPUT_SIZE


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = pil_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    base_arr = np.array(base, dtype=np.float32) / 255.0

    heat = cm.jet(cam)[..., :3]  # (H,W,3) RGB in [0,1]
    blended = (1 - alpha) * base_arr + alpha * heat
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)
