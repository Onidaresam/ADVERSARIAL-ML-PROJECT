# -*- coding: utf-8 -*-
"""
train_mlp_v3_all.py

Row-matched, scientifically accurate sensory comparison for MLP:
- Split indices ONCE and reuse for clean/poisoned.
- Sensory mask supports entry-wise and row-wise.

GPU if available; otherwise CPU (multi-threaded).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ------------------------------------------------------------------
# Drive sync helper
# ------------------------------------------------------------------
try:
    from scripts.gdrive_sync import ensure_local_tree
except Exception as e:
    print("[WARN] gdrive_sync import failed:", repr(e), flush=True)
    def ensure_local_tree(*, folder_id=None, cache_root=r"C:\Data\project_cache",
                          client_secret_path=None, **kwargs):
        return cache_root

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=str, required=True)
parser.add_argument("--attack", type=str, required=True,
                    choices=["label_flip", "sensory_add1"])
parser.add_argument("--segment", type=str, default="01")

parser.add_argument("--folder_id", type=str,
                    default="1bg1W_xGKiP5z0zM1CBIo7u6KbOm-1U0O")
parser.add_argument("--cache_root", type=str,
                    default=r"C:\Data\project_cache")
parser.add_argument("--client_secret", type=str,
                    default=r"C:\Dev\gpu\client_secret.json")

parser.add_argument("--sensory_mask_scope",
                    choices=["entry", "row"],
                    default="entry")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

PCT = args.pct
ATTACK = args.attack
SEGMENT_ID = args.segment
RANDOM_STATE = int(args.seed)

# ============================================================
# PATH CONFIG (Drive cache + local)
# ============================================================
def get_base_path():
    if ensure_local_tree is not None:
        data_root = ensure_local_tree(folder_id=args.folder_id,
                                      cache_root=args.cache_root,
                                      client_secret_path=args.client_secret)
        return str(Path(data_root) / "data")
    return str(Path(__file__).resolve().parents[2] / "data")

BASE = get_base_path()
SEG_FOLDER = f"{BASE}/segments_time/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

#SENSORY_FOLDER = f"{SEG_FOLDER}/sensory_poison_v3"
SENSORY_FOLDER = f"{SEG_FOLDER}/sensory_poison_v4"
LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips_v3"

#OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_v3"
#DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_v3_details"
#TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_mlp_v3"

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_v4"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_v4_details"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_mlp_v4"


os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)

print(f"\n[train_mlp_v3_all] Segment {SEGMENT_ID}")
print(f"Attack: {ATTACK}, Pct: {PCT}")
print(f"BASE: {BASE}")
print(f"Device: {device}")
print(f"Sensory mask scope: {args.sensory_mask_scope}")
print(f"Random seed: {RANDOM_STATE}\n")

# ============================================================
# SEEDING
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)

# ============================================================
# HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 256
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 512
NUM_BLOCKS = 4
SE_REDUCTION = 8

EARLY_STOP_PATIENCE = 12
FOCAL_GAMMA = 2.0
CB_BETA = 0.9999
WARMUP_EPOCHS = 5

# ============================================================
# DATA LOADING
# ============================================================
def load_clean():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "label"]
    X = df[numeric_cols].astype(float).values
    y = pd.read_csv(label_file).values
    return X, y

def load_poisoned(attack, pct):
    if attack == "label_flip":
        y_poison = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_v3_{pct}.csv"
        ).values
        X_clean, _ = load_clean()
        return X_clean, y_poison
    elif attack == "sensory_add1":
        pattern = f"RSSI_continuous_p{pct}_v3.csv"
        full_path = os.path.join(SENSORY_FOLDER, pattern)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Expected sensory file not found: {full_path}")
        dfp = pd.read_csv(full_path).astype(float)
        Xp = dfp.iloc[:, :-1].values
        y_clean = pd.read_csv(label_file).values
        return Xp, y_clean
    else:
        raise ValueError("Unsupported attack type")

# ============================================================
# DATASET
# ============================================================
class MultiLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# CLASS-BALANCED FOCAL LOSS
# ============================================================
class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=CB_BETA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.gamma = gamma

        n = samples_per_class.astype(float)
        n = np.clip(n, 1.0, None)
        effective_num = 1.0 - np.power(beta, n)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(weights)

        self.class_weights = torch.from_numpy(weights).float().to(device)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - pt) ** self.gamma
        cw = self.class_weights.unsqueeze(0)
        loss = focal_factor * bce * cw
        return loss.mean()

# ============================================================
# OVERSAMPLING
# ============================================================
def compute_sample_weights(y_train):
    pos_counts = y_train.sum(axis=1)
    return 1.0 + 2.0 * (pos_counts > 0).astype(float)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, drop_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.ln2(out)
        if self.drop_prob > 0 and self.training:
            if torch.rand(1).item() < self.drop_prob:
                return residual
        return self.act(out + residual)

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=SE_REDUCTION):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = x.mean(dim=0, keepdim=True)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        blocks = []
        for i in range(num_blocks):
            drop_prob = 0.05 * (i / max(1, num_blocks - 1))
            blocks.append(ResidualBlock(hidden_dim, hidden_dim * 2, drop_prob))
        self.blocks = nn.Sequential(*blocks)
        self.se = SEBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.se(x)
        return self.output_layer(x)

# ============================================================
# LR SCHEDULER
# ============================================================
def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=WARMUP_EPOCHS):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader, criterion, max_epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer, max_epochs)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{max_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model

# ============================================================
# PREDICTION
# ============================================================
def predict_model(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs >= 0.5).astype(int))
    return np.vstack(preds)

# ============================================================
# ASR / ASenR
# ============================================================
def compute_asr_label_flip(pred_poison, y_clean, y_poison):
    flipped_mask = (y_clean != y_poison)
    denom = flipped_mask.sum()
    if denom == 0:
        return 0.0, flipped_mask
    success = (pred_poison == y_poison) & flipped_mask
    return success.sum() / denom, flipped_mask

def _align_mask(mask, shape):
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        m = m[:, None]
        if shape[1] > 1:
            m = np.broadcast_to(m, shape)
    elif m.ndim == 2:
        if m.shape != shape:
            raise ValueError(f"Mask shape {m.shape} != predictions shape {shape}")
    else:
        raise ValueError("Mask must be 1-D or 2-D.")
    return m

def compute_asr_sensory(pred_clean, pred_poison, mask):
    pred_clean  = np.asarray(pred_clean)
    pred_poison = np.asarray(pred_poison)
    assert pred_clean.shape == pred_poison.shape
    m = _align_mask(mask, pred_clean.shape)
    changed = (pred_clean != pred_poison) & m
    denom = m.sum()
    return float(changed.sum() / denom) if denom else 0.0

def compute_asenr(pred_clean, pred_poison, mask):
    pred_clean  = np.asarray(pred_clean)
    pred_poison = np.asarray(pred_poison)
    assert pred_clean.shape == pred_poison.shape
    m = _align_mask(mask, pred_clean.shape)
    clean_positions = ~m
    denom = clean_positions.sum()
    if denom == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & clean_positions
    return float(changed.sum() / denom)

# ============================================================
# PER-BIN METRICS
# ============================================================
def compute_per_bin_metrics(y_true, y_pred, phase_name, attack=None, pct=None):
    rows = []
    rows_conf = []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        rows.append({
            "bin_index": j,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        rows_conf.append({
            "bin_index": j,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        })

    suffix = phase_name
    if attack is not None:
        suffix = f"{phase_name}_{attack}_{pct}"

    df_metrics = pd.DataFrame(rows)
    df_conf = pd.DataFrame(rows_conf)

    df_metrics.to_csv(f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_metrics.csv", index=False)
    df_conf.to_csv(f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_confusion.csv", index=False)

    return df_metrics

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    global_start = time.time()

    # LOAD CLEAN + POISONED
    X_clean, y_clean = load_clean()
    Xp, yp = load_poisoned(ATTACK, PCT)

    assert Xp.shape[0] == X_clean.shape[0]
    assert y_clean.shape[0] == X_clean.shape[0]
    N = X_clean.shape[0]

    all_idx = np.arange(N)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )

    X_train, X_test = X_clean[idx_train], X_clean[idx_test]
    y_train, y_test = y_clean[idx_train], y_clean[idx_test]

    Xp_train, Xp_test = Xp[idx_train], Xp[idx_test]
    yp_train, yp_test = yp[idx_train], yp[idx_test]

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest.npy", y_test)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # BASELINE TRAINING
    print("\n[BASELINE] Training MLP on CLEAN data...")

    samples_per_class = y_train.sum(axis=0)
    cb_focal = CBFocalLoss(samples_per_class)

    sample_weights = compute_sample_weights(y_train)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = MultiLabelDataset(X_train, y_train)
    val_ds = MultiLabelDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model_clean = MLP(input_dim, output_dim).to(device)
    model_clean = train_model(model_clean, train_loader, val_loader, cb_focal)

    preds_clean = predict_model(model_clean, X_test)

    compute_per_entry_metrics(
        y_test, y_test,
        preds_clean, preds_clean,
        poisoned_mask=np.zeros_like(preds_clean),  # no poisoning
        phase_name="baseline"
    )

    
    metrics_clean = {
        "accuracy_full": accuracy_score(y_test, preds_clean),
        "precision_full": precision_score(y_test, preds_clean, average="macro", zero_division=0),
        "recall_full": recall_score(y_test, preds_clean, average="macro", zero_division=0),
        "f1_full": f1_score(y_test, preds_clean, average="macro", zero_division=0),
    }

    df_bins_clean = compute_per_bin_metrics(
        y_test, preds_clean, phase_name="baseline"
    )

    metrics_clean["mean_perbin_accuracy"] = df_bins_clean["accuracy"].mean()
    metrics_clean["mean_perbin_f1"] = df_bins_clean["f1"].mean()

    pd.DataFrame([metrics_clean]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_mlp_clean_metrics.csv", index=False
    )

    print("Baseline metrics saved.")

    # ROBUSTNESS
    print(f"\n[ROBUSTNESS] Evaluating clean-trained MLP → {ATTACK} {PCT}%")

    preds_robust = predict_model(model_clean, Xp_test)

    compute_per_entry_metrics(
        y_test, yp_test,
        preds_clean, preds_robust,
        poisoned_mask,
        phase_name="robust",
        attack=ATTACK,
        pct=PCT
    )

    print("\n[DEBUG ROBUST] sensory check")
    print("  Xp_test != X_test entries:", (Xp_test != X_test).sum())
    print("  any poisoned rows:", np.any(Xp_test != X_test, axis=1).sum())
    print("  preds_clean != preds_robust entries:", (preds_clean != preds_robust).sum())

    
    if ATTACK == "label_flip":
        asr, flipped_mask = compute_asr_label_flip(preds_robust, y_clean=y_test, y_poison=yp_test)
        asenr = compute_asenr(preds_clean, preds_robust, flipped_mask)
    else:
        if args.sensory_mask_scope == "entry":
            poisoned_mask = (Xp_test != X_test)
        else:
            poisoned_mask = np.any(Xp_test != X_test, axis=1)
        asr = compute_asr_sensory(preds_clean, preds_robust, poisoned_mask)
        asenr = compute_asenr(preds_clean, preds_robust, poisoned_mask)

    metrics_robust = {
        "accuracy_full": accuracy_score(yp_test, preds_robust),
        "precision_full": precision_score(yp_test, preds_robust, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test, preds_robust, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test, preds_robust, average="macro", zero_division=0),
        "ASR": asr,
        "ASenR": asenr,
    }

    df_bins_robust = compute_per_bin_metrics(
        yp_test, preds_robust, phase_name="robust", attack=ATTACK, pct=PCT
    )




    metrics_robust["mean_perbin_accuracy"] = df_bins_robust["accuracy"].mean()
    metrics_robust["mean_perbin_f1"] = df_bins_robust["f1"].mean()

    pd.DataFrame([metrics_robust]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_mlp_robust_{ATTACK}_{PCT}_metrics.csv", index=False
    )

    print("Robustness metrics saved.")

    # ADVERSARIAL TRAINING
    print(f"\n[ADV TRAIN] Training MLP on {ATTACK} {PCT}% poisoned data...")

    samples_per_class_adv = yp_train.sum(axis=0)
    cb_focal_adv = CBFocalLoss(samples_per_class_adv)

    sample_weights_adv = compute_sample_weights(yp_train)
    sampler_adv = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights_adv).float(),
        num_samples=len(sample_weights_adv),
        replacement=True,
    )

    train_ds_adv = MultiLabelDataset(Xp_train, yp_train)
    val_ds_adv = MultiLabelDataset(Xp_test, yp_test)

    train_loader_adv = DataLoader(train_ds_adv, batch_size=BATCH_SIZE, sampler=sampler_adv)
    val_loader_adv = DataLoader(val_ds_adv, batch_size=BATCH_SIZE, shuffle=False)

    model_adv = MLP(input_dim, output_dim).to(device)
    model_adv = train_model(model_adv, train_loader_adv, val_loader_adv, cb_focal_adv)

    preds_adv = predict_model(model_adv, Xp_test)

    compute_per_entry_metrics(
        y_test, yp_test,
        preds_clean, preds_adv,
        poisoned_mask_adv,
        phase_name="advtrain",
        attack=ATTACK,
        pct=PCT
    )

    
    print("\n[DEBUG ADVTRAIN] sensory check")
    print("  Xp_test != X_test entries:", (Xp_test != X_test).sum())
    print("  any poisoned rows:", np.any(Xp_test != X_test, axis=1).sum())
    print("  preds_clean != preds_adv entries:", (preds_clean != preds_adv).sum())

    if ATTACK == "label_flip":
        asr_adv, flipped_mask_adv = compute_asr_label_flip(
            preds_adv, y_clean=yp_test, y_poison=yp_test
        )
        asenr_adv = compute_asenr(preds_clean, preds_adv, flipped_mask_adv)
    else:
        if args.sensory_mask_scope == "entry":
            poisoned_mask_adv = np.abs(Xp_test - X_test) > 1e-6
        else:
            poisoned_mask_adv = np.any(Xp_test != X_test, axis=1)
        asr_adv = compute_asr_sensory(preds_clean, preds_adv, poisoned_mask_adv)
        asenr_adv = compute_asenr(preds_clean, preds_adv, poisoned_mask_adv)

    metrics_adv = {
        "accuracy_full": accuracy_score(yp_test, preds_adv),
        "precision_full": precision_score(yp_test, preds_adv, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test, preds_adv, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test, preds_adv, average="macro", zero_division=0),
        "ASR": asr_adv,
        "ASenR": asenr_adv,
    }

    df_bins_adv = compute_per_bin_metrics(
        yp_test, preds_adv, phase_name="advtrain", attack=ATTACK, pct=PCT
    )

    metrics_adv["mean_perbin_accuracy"] = df_bins_adv["accuracy"].mean()
    metrics_adv["mean_perbin_f1"] = df_bins_adv["f1"].mean()

    pd.DataFrame([metrics_adv]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_mlp_advtrain_{ATTACK}_{PCT}_metrics.csv", index=False
    )

    print("Adversarial training metrics saved.\n")

    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("MLP v3 all pipeline completed.")
