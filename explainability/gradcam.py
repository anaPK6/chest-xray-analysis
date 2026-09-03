from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import DEVICE, INPUT_SIZE


class GradCAM:
    def __init__(self, model: torch.nn.Module, device: torch.device = DEVICE):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        # Hook the last dense block. Both torchxrayvision and torchvision
        # DenseNet121 expose `features.denseblock4`. We deliberately hook the
        # block (not `model.features`) because an in-place F.relu follows the
        # features container, and a backward hook on that view is forbidden.
        target_layer = self._find_target_layer(model)
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
        features = getattr(model, "features", None)
        if features is not None and hasattr(features, "denseblock4"):
            return features.denseblock4
        raise AttributeError(
            "Grad-CAM could not find features.denseblock4 on this model; "
            "pass a DenseNet121 or adjust the target layer."
        )

    def _save_activations(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        """img: (C,224,224) — C=1 (xrv) or C=3 (imagenet). -> (224,224) in [0,1]."""
        self.model.eval()
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        # weight each activation map by its mean gradient (channel importance).
        # ReLU the activations here since we hook before the model's own relu.
        acts = F.relu(self.activations)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze()
        return cam.cpu().numpy()
