import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Model: Logistic Regression (PyTorch, multi-label)
# =========================
class TorchLogReg(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)  # logits


# =========================
# Data utilities
# =========================
def load_bin_info(bin_info_folder):
    variable_bins = np.load(f"{bin_info_folder}/variable_bins.npy").astype(int)
    const_zero = np.load(f"{bin_info_folder}/constant_zero_bins.npy").astype(int)
    const_one = np.load(f"{bin_info_folder}/constant_one_bins.npy").astype(int)

    print("\n=== BIN INFO ===")
    print(f"Total bins (original): {len(variable_bins) + len(const_zero) + len(const_one)}")
    print(f"Variable bins:         {len(variable_bins)}")
    print(f"Constant-zero bins:    {len(const_zero)}")
    print(f"Constant-one bins:     {len(const_one)}\n")

    return variable_bins


def load_clean_data(data_folder, segment_id, variable_bins):
    t0 = time.time()
    X = pd.read_csv(f"{data_folder}/Cleaned_Data_{segment_id}.csv").values.astype("float32")
    Y = pd.read_csv(f"{data_folder}/Label_Matrix_{segment_id}.csv").values.astype("float32")

    X = X[:, variable_bins]

    print(f"[Time] Loaded clean data in {time.time() - t0:.2f}s")
    print(f"[Shape] X: {X.shape}, Y: {Y.shape}")
    return X, Y


def load_poisoned_data(data_folder, segment_id, variable_bins, attack, pct):
    t0 = time.time()
    poison_folder = f"{data_folder}/poisoned/{attack}/pct_{pct}"
    Xp = pd.read_csv(f"{poison_folder}/Poisoned_Data_{segment_id}.csv").values.astype("float32")
    Yp = pd.read_csv(f"{poison_folder}/Poisoned_Labels_{segment_id}.csv").values.astype("float32")

    Xp = Xp[:, variable_bins]

    print(f"[Time] Loaded poisoned data in {time.time() - t0:.2f}s")
    print(f"[Shape] Xp: {Xp.shape}, Yp: {Yp.shape}")
    return Xp, Yp


def make_dataloader(X, Y, batch_size=4096, shuffle=True):
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    ds = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# =========================
# Training / evaluation
# =========================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, Yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_targets = []

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        logits = model(Xb)
        all_logits.append(logits.cpu())
        all_targets.append(Yb.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    correct_per_label = (preds == targets).float().mean(dim=0).numpy()
    overall = (preds == targets).float().mean().item()

    return overall, correct_per_label


def save_results(path, info_dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info_dict, f, indent=2)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=str, required=True, help="Segment ID, e.g. 01")
    parser.add_argument("--pct", type=float, required=True, help="Poisoning percentage, e.g. 0.5")
    parser.add_argument("--attack", type=str, required=True, help="Attack type, e.g. label_flip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    segment_id = args.segment
    pct = args.pct
    attack = args.attack

    base_data_folder = f"data/segments_time/segment_{segment_id}"
    bin_info_folder = f"{base_data_folder}/models_reducedscope/bin_info"
    results_folder = f"results/segment_{segment_id}/logreg_torch"

    os.makedirs(results_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}\n")

    # ===== BIN INFO =====
    variable_bins = load_bin_info(bin_info_folder)

    # ===== CLEAN DATA =====
    print(f"[BASELINE] Training Torch Logistic Regression on CLEAN data (segment {segment_id}).")
    X_clean, Y_clean = load_clean_data(base_data_folder, segment_id, variable_bins)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_clean, Y_clean, test_size=0.2, random_state=42, shuffle=True
    )

    n_features = X_train.shape[1]
    n_labels = Y_train.shape[1]

    model_clean = TorchLogReg(in_features=n_features, out_features=n_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_clean = torch.optim.AdamW(model_clean.parameters(), lr=args.lr)

    train_loader_clean = make_dataloader(X_train, Y_train, batch_size=args.batch_size, shuffle=True)
    eval_loader_clean = make_dataloader(X_test, Y_test, batch_size=args.batch_size, shuffle=False)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model_clean, train_loader_clean, criterion, optimizer_clean, device)
        print(f"[CLEAN][Epoch {epoch}/{args.epochs}] Loss: {loss:.4f}")
    clean_train_time = time.time() - t0
    print(f"[CLEAN] Training time: {clean_train_time:.2f}s")

    clean_overall_acc, clean_per_label_acc = evaluate(model_clean, eval_loader_clean, device)
    print(f"[CLEAN] Overall accuracy (all labels): {clean_overall_acc:.4f}")

    # ===== POISONED DATA =====
    print(f"\n[POISON] Attack: {attack}, pct: {pct}% (segment {segment_id})")
    X_poison, Y_poison = load_poisoned_data(base_data_folder, segment_id, variable_bins, attack, pct)

    Xp_train, Xp_test, Yp_train, Yp_test = train_test_split(
        X_poison, Y_poison, test_size=0.2, random_state=42, shuffle=True
    )

    model_poison = TorchLogReg(in_features=n_features, out_features=n_labels).to(device)
    optimizer_poison = torch.optim.AdamW(model_poison.parameters(), lr=args.lr)

    train_loader_poison = make_dataloader(Xp_train, Yp_train, batch_size=args.batch_size, shuffle=True)
    eval_loader_poison = make_dataloader(Xp_test, Yp_test, batch_size=args.batch_size, shuffle=False)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_p = train_epoch(model_poison, train_loader_poison, criterion, optimizer_poison, device)
        print(f"[POISON][Epoch {epoch}/{args.epochs}] Loss: {loss_p:.4f}")
    poison_train_time = time.time() - t0
    print(f"[POISON] Training time: {poison_train_time:.2f}s")

    poison_overall_acc, poison_per_label_acc = evaluate(model_poison, eval_loader_poison, device)
    print(f"[POISON] Overall accuracy (all labels): {poison_overall_acc:.4f}")

    # ===== SAVE RESULTS =====
    result_path = Path(results_folder) / f"logreg_torch_segment_{segment_id}_{attack}_pct_{pct}.json"
    results = {
        "segment": segment_id,
        "attack": attack,
        "pct": pct,
        "n_features": int(n_features),
        "n_labels": int(n_labels),
        "clean_overall_acc": clean_overall_acc,
        "poison_overall_acc": poison_overall_acc,
        "clean_train_time_sec": clean_train_time,
        "poison_train_time_sec": poison_train_time,
    }
    save_results(result_path, results)
    print(f"\n[RESULTS] Saved to: {result_path}")


if __name__ == "__main__":
    main()
