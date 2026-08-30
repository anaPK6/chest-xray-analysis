"""Multi-label thoracic disease classifier (Phase 1).

Two interchangeable weight sources, selected by config.MODEL_SOURCE:

  "pretrained"  torchxrayvision DenseNet — v1, works with no training.
  "local"       your Kaggle-trained model, loaded via the handoff contract:
                  models/checkpoints/
                    state_dict.pth      trained DenseNet121 weights
                    labels.json         label order (list[str])
                    model_config.json   {arch, input_size, num_classes,
                                          norm: "xrv"|"imagenet", sigmoid: bool}

Both sources expose the SAME interface: .labels and .predict(img) -> {label: prob}.
Downstream (findings, Grad-CAM, report) never knows which one is loaded.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torchxrayvision as xrv

from utils.config import (
    CHECKPOINT_DIR,
    DEVICE,
    MODEL_SOURCE,
    PRETRAINED_WEIGHTS,
)


def _build_densenet121(num_classes: int) -> torch.nn.Module:
    """A plain torchvision DenseNet121 head sized to num_classes.

    Kaggle training builds the model the same way so the state_dict matches.
    """
    from torchvision.models import densenet121

    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


class XRayClassifier:
    def __init__(self, source: str = MODEL_SOURCE, device: torch.device = DEVICE):
        self.device = device
        self.source = source
        if source == "pretrained":
            self._load_pretrained()
        elif source == "local":
            self._load_local()
        else:
            raise ValueError(f"unknown MODEL_SOURCE: {source!r}")
        self.model = self.model.to(device).eval()

    # -- loaders --------------------------------------------------------------
    def _load_pretrained(self):
        self.model = xrv.models.DenseNet(weights=PRETRAINED_WEIGHTS)
        self.labels = list(self.model.pathologies)
        self._norm = "xrv"        # torchxrayvision's own forward applies sigmoid
        self._sigmoid = False     # so we do NOT sigmoid again
        self._xrv_native = True
        self._cfg = {}

    def _load_local(self):
        cfg_path = CHECKPOINT_DIR / "model_config.json"
        labels_path = CHECKPOINT_DIR / "labels.json"
        weights_path = CHECKPOINT_DIR / "state_dict.pth"
        for p in (cfg_path, labels_path, weights_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"MODEL_SOURCE=local but missing {p.name} in {CHECKPOINT_DIR}. "
                    "Export it from the Kaggle training notebook."
                )
        cfg = json.loads(cfg_path.read_text())
        self.labels = json.loads(labels_path.read_text())
        assert cfg.get("num_classes", len(self.labels)) == len(self.labels), (
            "num_classes in model_config.json does not match labels.json"
        )
        self.model = _build_densenet121(len(self.labels))
        state = torch.load(weights_path, map_location="cpu")
        # tolerate checkpoints saved as {"model": state_dict} or raw state_dict
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state)
        self._norm = cfg.get("norm", "imagenet")
        self._sigmoid = cfg.get("sigmoid", True)  # our head outputs logits
        self._xrv_native = False
        self._cfg = cfg

    @property
    def norm(self) -> str:
        """Normalization this model expects; drives utils.preprocess."""
        return self._norm

    def model_info(self) -> dict:
        """Summary for the UI: which model, how it was trained, its scores."""
        return {
            "source": self.source,
            "arch": self._cfg.get("arch", "densenet121"),
            "labels": [l for l in self.labels if l],
            "trained_on": self._cfg.get("trained_on"),
            "epochs": self._cfg.get("epochs"),
            "mean_auroc": self._cfg.get("mean_auroc"),
            "per_label_auroc": self._cfg.get("per_label_auroc"),
            "pretrained_weights": PRETRAINED_WEIGHTS if self.source == "pretrained" else None,
        }

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, img: np.ndarray) -> dict[str, float]:
        """img: preprocessed array — (1,H,W) for xrv or (3,H,W) for imagenet.

        Returns {pathology: prob}.
        """
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)  # add batch dim
        out = self.model(x)[0]
        if self._sigmoid:
            out = torch.sigmoid(out)
        out = out.cpu().numpy()
        return {
            label: float(p)
            for label, p in zip(self.labels, out)
            if label  # skip empty label slots (pretrained has some)
        }
