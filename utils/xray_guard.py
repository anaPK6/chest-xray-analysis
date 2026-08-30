"""Lightweight 'is this a frontal chest X-ray?' guard.

Public uploads may be selfies, memes, colour photos, or the wrong body part.
The classifier would happily emit disease probabilities for any of them, which
looks broken. This module scores how X-ray-like an image is from a few cheap,
transparent signals — no model, no downloads — so the UI can warn the user.

It is deliberately permissive (warn, don't block): the goal is to catch obvious
non-radiographs, not to be a precise classifier.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# below this combined score we consider it "probably not a chest X-ray"
XRAY_SCORE_THRESHOLD = 0.5


def _grayscale_score(rgb: np.ndarray) -> float:
    """1.0 only if pixels are ESSENTIALLY monochrome (R≈G≈B everywhere).

    Real radiographs are true grayscale (per-pixel channel std ~0). Photos —
    even muted/neutral ones — carry small but consistent colour variance. We
    penalize hard so a neutral-toned photo (grey building, cream clothing) still
    fails, not just saturated ones.
    """
    # mean per-pixel spread across R/G/B channels, in 0..255
    chan_std = rgb.std(axis=2).mean()
    # true X-ray: chan_std ≈ 0-1. Photo: typically > 3 even when muted.
    # steep penalty: ~1.0 at 0, hits 0 around chan_std ≈ 3.5
    return float(np.clip(1.0 - chan_std / 3.5, 0.0, 1.0))


def _histogram_score(gray: np.ndarray) -> float:
    """Radiographs span a broad range (dark lungs → bright bone) without being
    dominated by pure black or white. Reward good spread, penalize clipping."""
    g = gray / 255.0
    spread = g.std()                       # broad tonal range expected
    # near-flat images (blank banners, solid fills) are never radiographs
    if spread < 0.06:
        return 0.0
    spread_s = np.clip(spread / 0.22, 0, 1)
    # fraction of near-pure-black/white pixels (memes/screenshots clip hard)
    extreme = np.mean((g < 0.02) | (g > 0.98))
    extreme_s = np.clip(1.0 - extreme * 3.0, 0, 1)
    return float(spread_s * 0.6 + extreme_s * 0.4)


def _aspect_score(w: int, h: int) -> float:
    """Chest films are roughly square-to-portrait. Penalize very wide/tall."""
    ar = w / h if h else 1.0
    # ideal ~0.8-1.1; decay outside 0.6-1.4
    if 0.7 <= ar <= 1.15:
        return 1.0
    if ar < 0.7:
        return float(np.clip(ar / 0.7, 0, 1))
    return float(np.clip(1.4 / ar, 0, 1))


def xray_likeness(pil_img: Image.Image) -> dict:
    """Return {is_xray, score, reasons, parts} for an uploaded image.

    Grayscale-ness is a HARD GATE: a real chest X-ray is essentially monochrome,
    so any image with meaningful colour is rejected outright regardless of the
    other signals. This is what stops photos (colourful OR neutral-toned) from
    passing — a photo always carries some per-pixel colour variance.
    """
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    w, h = pil_img.size

    gs = _grayscale_score(rgb)
    hs = _histogram_score(gray)
    asp = _aspect_score(w, h)

    reasons = []
    # HARD GATE: not near-monochrome -> definitely a photo, not an X-ray.
    if gs < 0.85:
        reasons.append("image contains colour — chest X-rays are grayscale")
        return {
            "is_xray": False,
            "score": round(float(0.3 * gs), 3),
            "reasons": reasons,
            "parts": {"grayscale": round(gs, 3), "histogram": round(hs, 3), "aspect": round(asp, 3)},
        }

    # past the gate: it's monochrome — judge structure (histogram/aspect).
    score = 0.5 * gs + 0.3 * hs + 0.2 * asp
    if hs < 0.4:
        reasons.append("intensity distribution is unlike a radiograph")
    if asp < 0.5:
        reasons.append("unusual aspect ratio for a frontal chest film")

    return {
        "is_xray": score >= XRAY_SCORE_THRESHOLD,
        "score": round(float(score), 3),
        "reasons": reasons,
        "parts": {"grayscale": round(gs, 3), "histogram": round(hs, 3), "aspect": round(asp, 3)},
    }
