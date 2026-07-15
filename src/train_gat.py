"""
train_gat.py
============
End-to-end training pipeline for the Browser-GAT anomaly classifier.

Splits by session_id (NOT random row split), trains with early stopping,
and reports accuracy, precision, recall, F1 and ROC-AUC on the held-out
test set.

Usage
-----
    python train_gat.py [--root data] [--epochs 100] [--lr 1e-3]

The trained model is saved to ``models/browser_gat.pt``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from gat_model import BrowserGAT
from pyg_dataset import BrowserSessionDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Session-stratified split
# ---------------------------------------------------------------------------

def session_split(
    dataset: BrowserSessionDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split ``dataset`` by *session*, not by row, to prevent label leakage.

    Returns
    -------
    train_indices, val_indices, test_indices
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))

    # Separate normal and malicious to ensure balanced splits
    normal_idx    = [i for i in indices if dataset[i].y.item() == 0]
    malicious_idx = [i for i in indices if dataset[i].y.item() == 1]

    def _split(lst: list[int]) -> tuple[list[int], list[int], list[int]]:
        rng.shuffle(lst)
        n = len(lst)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(0, int(n * val_ratio))
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    tr_n, va_n, te_n = _split(normal_idx)
    tr_m, va_m, te_m = _split(malicious_idx)

    return tr_n + tr_m, va_n + va_m, te_n + te_m


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BrowserGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: BrowserGAT,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    model.eval()
    all_labels  = []
    all_preds   = []
    all_probs   = []
    total_loss  = 0.0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss   = F.cross_entropy(logits, batch.y, weight=class_weights)
        total_loss += loss.item() * batch.num_graphs
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    metrics: dict[str, float] = {
        "loss":      total_loss / max(len(loader.dataset), 1),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }
    if len(np.unique(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    root: str = "data",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    hidden_channels: int = 64,
    num_heads: int = 4,
    dropout: float = 0.4,
    patience: int = 15,
    seed: int = 42,
    model_path: str = "models/browser_gat.pt",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset --------------------------------------------------------- #
    print("\nLoading dataset …")
    dataset = BrowserSessionDataset(root=root)
    print(f"Total graphs: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: no graphs — check that raw JSON logs are in data/raw/")
        return

    # ---- Split ----------------------------------------------------------- #
    train_idx, val_idx, test_idx = session_split(dataset, seed=seed)
    print(
        f"Split → train: {len(train_idx)}  "
        f"val: {len(val_idx)}  "
        f"test: {len(test_idx)}"
    )

    train_ds = [dataset[i] for i in train_idx]
    val_ds   = [dataset[i] for i in val_idx]
    test_ds  = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # ---- Class weights (handle imbalance) -------------------------------- #
    labels = torch.tensor([dataset[i].y.item() for i in train_idx])
    n_neg  = (labels == 0).sum().item()
    n_pos  = (labels == 1).sum().item()
    if n_pos > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float).to(device)
    else:
        weight = None

    # ---- Model ----------------------------------------------------------- #
    in_channels = dataset[0].x.size(1)
    model = BrowserGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_classes=2,
        dropout=dropout,
    ).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5, min_lr=1e-5
    )

    # ---- Training loop --------------------------------------------------- #
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(Path(model_path).parent, exist_ok=True)

    print("\n" + "─" * 75)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'Val Acc':>9} {'Val F1':>9} {'Val AUC':>9}")
    print("─" * 75)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weight)
        val_metrics = evaluate(model, val_loader, device, weight)
        val_loss    = val_metrics["loss"]

        scheduler.step(val_loss)

        print(
            f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
            f"{val_metrics['accuracy']:>9.4f} {val_metrics['f1']:>9.4f} "
            f"{val_metrics['roc_auc']:>9.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Test evaluation ------------------------------------------------- #
    print("\n" + "─" * 75)
    print("Loading best model for test evaluation …")
    model = BrowserGAT.load(
        model_path,
        device=device,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_classes=2,
        dropout=dropout,
    )
    test_metrics = evaluate(model, test_loader, device)

    print("\nTest results:")
    print("─" * 40)
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("─" * 40)
    print(f"\nModel saved → {model_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Browser-GAT")
    parser.add_argument("--root",    default="data",               help="Dataset root directory")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--wd",      type=float, default=1e-4,     dest="weight_decay")
    parser.add_argument("--batch",   type=int,   default=32,       dest="batch_size")
    parser.add_argument("--hidden",  type=int,   default=64,       dest="hidden_channels")
    parser.add_argument("--heads",   type=int,   default=4,        dest="num_heads")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience",type=int,   default=15)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--model",   default="models/browser_gat.pt", dest="model_path")
    args = parser.parse_args()
    train(**vars(args))
