"""Local CheXpert training on Apple Silicon (MPS) — no cloud, no hanging sessions.

Trains DenseNet121 on a SUBSET of CheXpert-small and exports the handoff
contract (state_dict.pth + labels.json + model_config.json) into
models/checkpoints/ so the app can load it with MODEL_SOURCE=local.

Usage:
    # fast smoke-test (a few minutes) — proves the whole loop end to end
    python train_local.py --subset 1500 --epochs 1

    # a real portfolio run (a few hours on M4) — respectable AUROC
    python train_local.py --subset 15000 --epochs 2

Data is expected under ./data (downloaded via `kaggle datasets download -d ashery/chexpert`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models" / "checkpoints"

# 5 "competition" CheXpert labels most papers report AUROC on.
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# Uncertainty (-1) mapping: "ones" (U-Ones) helps Atelectasis/Edema; "zeros" else.
UNCERTAIN_POLICY = {
    "Atelectasis": "ones", "Cardiomegaly": "zeros", "Consolidation": "zeros",
    "Edema": "ones", "Pleural Effusion": "zeros",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_data_root() -> Path:
    """Locate the dir containing train.csv under ./data (any nesting)."""
    hits = glob.glob(str(DATA_DIR / "**" / "train.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"train.csv not found under {DATA_DIR}. Download CheXpert-small first:\n"
            "  kaggle datasets download -d ashery/chexpert -p data --unzip"
        )
    return Path(os.path.dirname(hits[0]))


def load_split(data_root: Path, csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(data_root / csv_name)
    for lab in LABELS:
        policy = UNCERTAIN_POLICY.get(lab, "zeros")
        col = df[lab].replace(-1.0, 1.0 if policy == "ones" else 0.0)
        df[lab] = col.fillna(0.0).clip(0, 1)
    return df


def resolve_image_path(img_root: Path, rel_path: str) -> Path:
    """CSV 'Path' may carry a 'CheXpert-v1.0-small/' prefix that isn't a real
    dir in this dataset (images unzip straight to data/train, data/valid).
    Try the path as-is, then with that prefix stripped."""
    p = img_root / rel_path
    if p.exists():
        return p
    parts = rel_path.split("/", 1)
    if len(parts) == 2 and parts[0].lower().startswith("chexpert"):
        return img_root / parts[1]
    return p  # let it raise a clear FileNotFoundError if truly missing


class CheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: Path, tf):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(resolve_image_path(self.img_root, row["Path"])).convert("RGB")
        y = torch.tensor([row[l] for l in LABELS], dtype=torch.float32)
        return self.tf(img), y


def build_model() -> torch.nn.Module:
    # Same construction the app's classifier uses, so the state_dict matches.
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    aucs = {}
    for i, lab in enumerate(LABELS):
        try:
            aucs[lab] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            aucs[lab] = float("nan")  # only one class present in valid split
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=1500, help="train images (None-like: use -1 for full)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    data_root = find_data_root()
    # Images live under data_root (train/, valid/ next to the CSVs). resolve_image_path
    # strips the CheXpert-v1.0-small/ prefix the CSV carries.
    img_root = data_root
    print(f"data_root: {data_root}")

    train_df = load_split(data_root, "train.csv")
    valid_df = load_split(data_root, "valid.csv")
    if args.subset and args.subset > 0:
        train_df = train_df.sample(n=min(args.subset, len(train_df)), random_state=0).reset_index(drop=True)
    print(f"train: {len(train_df)}  valid: {len(valid_df)}")

    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dl = DataLoader(CheXpertDataset(train_df, img_root, train_tf),
                          batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_dl = DataLoader(CheXpertDataset(valid_df, img_root, eval_tf),
                          batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running, t0 = 0.0, time.time()
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}")
        aucs = evaluate(model, valid_dl, device)
        mean_auc = np.nanmean(list(aucs.values()))
        print(f"== epoch {epoch}: loss {running/len(train_dl):.4f} | mean AUROC {mean_auc:.4f} "
              f"| {time.time()-t0:.0f}s")
        for lab, a in aucs.items():
            print(f"     {lab:20s} {a:.4f}")

    # --- export the handoff contract ---
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "state_dict.pth")
    (CKPT_DIR / "labels.json").write_text(json.dumps(LABELS, indent=2))
    (CKPT_DIR / "model_config.json").write_text(json.dumps({
        "arch": "densenet121", "input_size": INPUT_SIZE, "num_classes": len(LABELS),
        "norm": "imagenet", "sigmoid": True,
        "trained_on": f"CheXpert-small subset={len(train_df)}", "epochs": args.epochs,
        "mean_auroc": float(mean_auc),
        "per_label_auroc": {lab: float(a) for lab, a in aucs.items()},
        "uncertain_policy": UNCERTAIN_POLICY,
    }, indent=2))
    print(f"\nwrote weights to {CKPT_DIR}. Run the app with MODEL_SOURCE=local.")


if __name__ == "__main__":
    main()
