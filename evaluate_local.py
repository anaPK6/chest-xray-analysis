"""Evaluate the trained model on the CheXpert validation set and save the
results the Evaluation UI needs (ROC points, AUROC, confusion metrics, info).

Run once after training:  python evaluate_local.py
Writes: models/checkpoints/evaluation.json

Reuses train_local.py's data loading so labels/uncertainty/paths match exactly.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from torchvision import transforms

from train_local import (
    CKPT_DIR, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, LABELS,
    CheXpertDataset, build_model, find_data_root, get_device, load_split,
)

# same eval transform train_local uses (defined inside its main(), so rebuilt here)
eval_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0   # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": round(sens, 3), "specificity": round(spec, 3),
            "accuracy": round(acc, 3)}


def main():
    device = get_device()
    data_root = find_data_root()
    valid_df = load_split(data_root, "valid.csv")
    dl = DataLoader(CheXpertDataset(valid_df, data_root, eval_tf),
                    batch_size=16, shuffle=False, num_workers=4)

    # load the trained weights
    model = build_model().to(device).eval()
    state = torch.load(CKPT_DIR / "state_dict.pth", map_location="cpu")
    model.load_state_dict(state)

    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            ys.append(y.numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    print(f"evaluated {len(y_true)} validation images")

    cfg = json.loads((CKPT_DIR / "model_config.json").read_text())
    per_label = {}
    for i, lab in enumerate(LABELS):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            # subsample the curve to ~40 points to keep the JSON small
            idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
            roc = [[round(float(fpr[j]), 4), round(float(tpr[j]), 4)] for j in idx]
        except ValueError:
            auc, roc = float("nan"), []
        per_label[lab] = {
            "auroc": round(auc, 4),
            "roc": roc,  # list of [fpr, tpr]
            "confusion": confusion_at(y_true[:, i], y_prob[:, i]),
            "positives": int(y_true[:, i].sum()),
        }

    mean_auroc = float(np.nanmean([per_label[l]["auroc"] for l in LABELS]))
    out = {
        "n_valid": int(len(y_true)),
        "threshold": 0.5,
        "mean_auroc": round(mean_auroc, 4),
        "labels": LABELS,
        "per_label": per_label,
        "model": {
            "arch": cfg.get("arch", "densenet121"),
            "trained_on": cfg.get("trained_on"),
            "epochs": cfg.get("epochs"),
            "uncertain_policy": cfg.get("uncertain_policy"),
        },
    }
    (CKPT_DIR / "evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"mean AUROC {mean_auroc:.4f} -> wrote {CKPT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
