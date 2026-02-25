import os
import argparse
import time
import random
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

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=str, required=True)
parser.add_argument(
    "--attack",
    type=str,
    required=True,
    choices=["label_flip", "sensory_add1"],
)
args = parser.parse_args()

PCT = args.pct
ATTACK = args.attack

# ============================================================
# CONFIG
# ============================================================
SEGMENT_ID = "01"
SEGMENT_TYPE = "time"

BASE = "data"

SEG_FOLDER = (
    f"{BASE}/segments_time/segment_{SEGMENT_ID}"
    if SEGMENT_TYPE == "time"
    else f"{BASE}/segments_random/random_{SEGMENT_ID}"
)

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

SENSORY_FOLDER = f"{SEG_FOLDER}/sensory_poison"
LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips"

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_cb_focal_research"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_cb_focal_research"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/mlp_cb_focal_research_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

set_seed(42)

# ============================================================
# LOAD BIN INFO
# ============================================================
variable_bins = np.load(f"{BIN_INFO_FOLDER}/variable_bins.npy").astype(int)
constant_zero_bins = np.load(f"{BIN_INFO_FOLDER}/constant_zero_bins.npy").astype(int)
constant_one_bins = np.load(f"{BIN_INFO_FOLDER}/constant_one_bins.npy").astype(int)

NUM_BINS = len(variable_bins) + len(constant_zero_bins) + len(constant_one_bins)

print("\n=== BIN INFO (ORIGINAL) ===")
print(f"Total bins (original): {NUM_BINS}")
print(f"Variable bins:         {len(variable_bins)}")
print(f"Constant-zero bins:    {len(constant_zero_bins)}")
print(f"Constant-one bins:     {len(constant_one_bins)}\n")

# ============================================================
# ORIGINAL OPTIMAL HYPERPARAMETERS (RESTORED)
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
MIN_POS_PER_BIN = 20

WARMUP_EPOCHS = 5

# ============================================================
# DATA LOADING
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(float).values
    y_full = pd.read_csv(label_file).values
    y_var = y_full[:, variable_bins]
    return X, y_full, y_var


def load_poisoned_features_labels(attack_type, pct):
    if attack_type == "label_flip":
        pct_str = str(pct).replace(".", "_")
        y_full = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_{pct_str}.csv"
        ).values
        X, _, _ = load_clean_features_labels()
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

    elif attack_type == "sensory_add1":
        pattern = f"RSSI_p{pct}"
        candidates = [
            f for f in os.listdir(SENSORY_FOLDER)
            if f.startswith(pattern) and f.endswith(".csv")
        ]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No sensory poisoning file found starting with: {pattern}"
            )
        sensory_file = candidates[0]
        X = pd.read_csv(os.path.join(SENSORY_FOLDER, sensory_file)).astype(float).values
        y_full = pd.read_csv(label_file).values
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

# ============================================================
# BIN FILTERING
# ============================================================
def filter_bins_by_positives(y_var, min_pos=MIN_POS_PER_BIN):
    pos_counts = y_var.sum(axis=0)
    keep_mask = pos_counts >= min_pos
    kept_indices = variable_bins[keep_mask]
    print(f"Filtering bins with < {min_pos} positives:")
    print(f"  Kept variable bins: {len(kept_indices)} / {len(variable_bins)}")
    return kept_indices, keep_mask

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
    pos_counts_per_sample = y_train.sum(axis=1)
    weights = 1.0 + 2.0 * (pos_counts_per_sample > 0).astype(float)
    return weights

# ============================================================
# MODEL: RESMLP + SE + LAYERNORM + GELU
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, drop_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerLayerNorm(dim)
        self.act = nn.GELU()
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.ln2(out)
        if self.drop_prob > 0.0 and self.training:
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
            blocks.append(ResidualBlock(hidden_dim, hidden_dim * 2, drop_prob=drop_prob))
        self.blocks = nn.Sequential(*blocks)

        self.se = SEBlock(hidden_dim, reduction=SE_REDUCTION)
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
# TRAIN / EVAL
# ============================================================
def train_model(model, train_loader, val_loader, criterion, max_epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer, max_epochs)

    best_val_loss = float("inf")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

        print(
            f"Epoch {epoch+1:03d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_model(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(X[i:i + BATCH_SIZE]).float().to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs >= 0.5).astype(int))
    return np.vstack(preds)
# ============================================================
# RECONSTRUCT FULL PREDICTIONS
# ============================================================
def reconstruct_full(y_var_pred, y_full_true, kept_var_bins):
    full_pred = np.zeros_like(y_full_true)
    full_pred[:, kept_var_bins] = y_var_pred

    if len(constant_zero_bins) > 0:
        full_pred[:, constant_zero_bins] = 0
    if len(constant_one_bins) > 0:
        full_pred[:, constant_one_bins] = 1

    return full_pred

# ============================================================
# POISON FLAG HELPER (same logic as LR)
# ============================================================
def compute_constant_poison_flags(y_full_true):
    """
    Returns an array of shape (N, NUM_BINS) with 1 where a constant bin
    deviates from its expected clean value, 0 otherwise.
    Variable bins are always 0 here.
    """
    N = y_full_true.shape[0]
    poison_flags = np.zeros_like(y_full_true, dtype=int)

    if len(constant_zero_bins) > 0:
        cz_true = y_full_true[:, constant_zero_bins]
        poison_flags[:, constant_zero_bins] = (cz_true != 0).astype(int)

    if len(constant_one_bins) > 0:
        co_true = y_full_true[:, constant_one_bins]
        poison_flags[:, constant_one_bins] = (co_true != 1).astype(int)

    return poison_flags

# ============================================================
# PER-BIN METRICS + GLOBAL SUMMARY
# ============================================================
def compute_per_bin_metrics(y_true, y_pred, kept_var_bins, phase_name, attack=None, pct=None):
    n_bins = y_true.shape[1]
    rows_metrics = []
    rows_conf = []

    for j in range(n_bins):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        rows_metrics.append({
            "bin_index": int(kept_var_bins[j]),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        rows_conf.append({
            "bin_index": int(kept_var_bins[j]),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })

    suffix = phase_name
    if attack is not None and pct is not None:
        suffix = f"{phase_name}_{attack}_{pct}"

    metrics_df = pd.DataFrame(rows_metrics)
    conf_df = pd.DataFrame(rows_conf)

    metrics_df.to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_metrics.csv",
        index=False
    )
    conf_df.to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_confusion.csv",
        index=False
    )

    return metrics_df


def summarize_per_bin_metrics(per_bin_df: pd.DataFrame) -> dict:
    summary = {}
    if "accuracy" in per_bin_df.columns:
        summary["mean_bin_accuracy"] = per_bin_df["accuracy"].mean()
    if "f1" in per_bin_df.columns:
        summary["mean_bin_f1"] = per_bin_df["f1"].mean()
    return summary

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning RESEARCH-GRADE CB-Focal MLP pipeline for segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}")
    print(f"Device: {device}\n")

    global_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    X_clean, y_clean_full, y_clean_var = load_clean_features_labels()

    kept_var_bins, keep_mask = filter_bins_by_positives(y_clean_var, MIN_POS_PER_BIN)
    y_clean_var = y_clean_var[:, keep_mask]

    X_train_full, X_test, y_train_full, y_test_full = train_test_split(
        X_clean, y_clean_full, test_size=0.2, shuffle=True
    )
    y_train_var_full = y_train_full[:, kept_var_bins]
    y_test_var = y_test_full[:, kept_var_bins]

    X_train, X_val, y_train_var, y_val_var = train_test_split(
        X_train_full, y_train_var_full, test_size=0.2, shuffle=True
    )

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest_full.npy", y_test_full)

    input_dim = X_train.shape[1]
    output_dim = y_train_var.shape[1]

    samples_per_class = y_train_var.sum(axis=0)
    cb_focal = CBFocalLoss(samples_per_class=samples_per_class)

    sample_weights = compute_sample_weights(y_train_var)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = MultiLabelDataset(X_train, y_train_var)
    val_ds = MultiLabelDataset(X_val, y_val_var)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ========================================================
    # BASELINE
    # ========================================================
    print("\n[BASELINE] Training RESEARCH-GRADE CB-Focal MLP on CLEAN data...")
    baseline_start = time.time()

    model_clean = MLP(input_dim, output_dim).to(device)
    model_clean = train_model(model_clean, train_loader, val_loader, cb_focal)

    baseline_end = time.time()
    print(f"[Time] Baseline training completed in {(baseline_end - baseline_start)/60:.2f} minutes")

    preds_var_clean = predict_model(model_clean, X_test)
    full_pred_clean = reconstruct_full(preds_var_clean, y_test_full, kept_var_bins)

    # === SAVE BASELINE PREDICTIONS FOR ASR/ASenR ===
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_clean_full_pred.npy", full_pred_clean)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_clean_ytest_full.npy", y_test_full)

    poison_flags_clean = compute_constant_poison_flags(y_test_full)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_clean_poison_flags.npy", poison_flags_clean)


    metrics_var_clean = {
        "accuracy_var": accuracy_score(y_test_var, preds_var_clean),
        "precision_var": precision_score(y_test_var, preds_var_clean, average="macro", zero_division=0),
        "recall_var": recall_score(y_test_var, preds_var_clean, average="macro", zero_division=0),
        "f1_var": f1_score(y_test_var, preds_var_clean, average="macro", zero_division=0),
    }

    metrics_full_clean = {
        "accuracy_full": accuracy_score(y_test_full, full_pred_clean),
        "precision_full": precision_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
        "recall_full": recall_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
        "f1_full": f1_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
    }

    # Combine var/full metrics
    baseline_metrics = {**metrics_var_clean, **metrics_full_clean}

    # Per-bin metrics
    per_bin_df_clean = compute_per_bin_metrics(
        y_test_var, preds_var_clean, kept_var_bins, phase_name="baseline"
    )

    # Global metrics (mean per-bin)
    baseline_summary = summarize_per_bin_metrics(per_bin_df_clean)

    # Write CSV with two rows
    baseline_csv_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_metrics.csv"
    df1 = pd.DataFrame([baseline_metrics])
    df2 = pd.DataFrame([baseline_summary])
    pd.concat([df1, df2], ignore_index=True).to_csv(baseline_csv_path, index=False)

    print("Baseline metrics saved.")

    # ========================================================
    # ROBUSTNESS
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained MLP → {ATTACK} {PCT}%")

    Xp, yp_full, yp_var = load_poisoned_features_labels(ATTACK, PCT)
    yp_var = yp_var[:, keep_mask]

    _, Xp_test, _, yp_test_full = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_test_var = yp_test_full[:, kept_var_bins]

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full)

    preds_var_robust = predict_model(model_clean, Xp_test)
    full_pred_robust = reconstruct_full(preds_var_robust, yp_test_full, kept_var_bins)

    # === SAVE ROBUSTNESS PREDICTIONS FOR ASR/ASenR ===
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_robust_{ATTACK}_{PCT}_full_pred.npy", full_pred_robust)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_robust_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full)

    poison_flags_robust = compute_constant_poison_flags(yp_test_full)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_robust_{ATTACK}_{PCT}_poison_flags.npy", poison_flags_robust)


    metrics_var_robust = {
        "accuracy_var": accuracy_score(yp_test_var, preds_var_robust),
        "precision_var": precision_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),
        "recall_var": recall_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),
        "f1_var": f1_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),
    }

    metrics_full_robust = {
        "accuracy_full": accuracy_score(yp_test_full, full_pred_robust),
        "precision_full": precision_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
    }

    robust_metrics = {**metrics_var_robust, **metrics_full_robust}

    per_bin_df_robust = compute_per_bin_metrics(
        yp_test_var,
        preds_var_robust,
        kept_var_bins,
        phase_name="robust",
        attack=ATTACK,
        pct=PCT
    )

    robust_summary = summarize_per_bin_metrics(per_bin_df_robust)

    robust_csv_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv"
    df1 = pd.DataFrame([robust_metrics])
    df2 = pd.DataFrame([robust_summary])
    pd.concat([df1, df2], ignore_index=True).to_csv(robust_csv_path, index=False)

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING
    # ========================================================
    print(f"\n[ADV TRAIN] Training MLP on {ATTACK} {PCT}% poisoned data...")

    Xp_train_full, Xp_test_adv, yp_train_full, yp_test_full_adv = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_train_var_full = yp_train_full[:, kept_var_bins]
    yp_test_var_adv = yp_test_full_adv[:, kept_var_bins]

    Xp_train, Xp_val, yp_train_var, yp_val_var = train_test_split(
        Xp_train_full, yp_train_var_full, test_size=0.2, shuffle=True
    )

    samples_per_class_adv = yp_train_var.sum(axis=0)
    cb_focal_adv = CBFocalLoss(samples_per_class=samples_per_class_adv)

    sample_weights_adv = compute_sample_weights(yp_train_var)
    sampler_adv = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights_adv).float(),
        num_samples=len(sample_weights_adv),
        replacement=True,
    )

    train_ds_adv = MultiLabelDataset(Xp_train, yp_train_var)
    val_ds_adv = MultiLabelDataset(Xp_val, yp_val_var)

    train_loader_adv = DataLoader(train_ds_adv, batch_size=BATCH_SIZE, sampler=sampler_adv)
    val_loader_adv = DataLoader(val_ds_adv, batch_size=BATCH_SIZE, shuffle=False)
    # ========================================================
    # ADVERSARIAL TRAINING (CONTINUED)
    # ========================================================
    adv_start = time.time()
    model_adv = MLP(input_dim, output_dim).to(device)
    model_adv = train_model(model_adv, train_loader_adv, val_loader_adv, cb_focal_adv)
    adv_end = time.time()

    print(f"[Time] Adversarial training completed in {(adv_end - adv_start)/60:.2f} minutes")

    # -----------------------------
    # ADVERSARIAL EVALUATION
    # -----------------------------
    preds_var_adv = predict_model(model_adv, Xp_test_adv)
    full_pred_adv = reconstruct_full(preds_var_adv, yp_test_full_adv, kept_var_bins)

    # === SAVE ADVTRAIN PREDICTIONS FOR ASR/ASenR ===
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_advtrain_{ATTACK}_{PCT}_full_pred.npy", full_pred_adv)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_advtrain_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full_adv)

    poison_flags_adv = compute_constant_poison_flags(yp_test_full_adv)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_mlp_advtrain_{ATTACK}_{PCT}_poison_flags.npy", poison_flags_adv)


    metrics_var_adv = {
        "accuracy_var": accuracy_score(yp_test_var_adv, preds_var_adv),
        "precision_var": precision_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "recall_var": recall_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "f1_var": f1_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
    }

    metrics_full_adv = {
        "accuracy_full": accuracy_score(yp_test_full_adv, full_pred_adv),
        "precision_full": precision_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
    }

    adv_metrics = {**metrics_var_adv, **metrics_full_adv}

    # Per-bin metrics
    per_bin_df_adv = compute_per_bin_metrics(
        yp_test_var_adv,
        preds_var_adv,
        kept_var_bins,
        phase_name="advtrain",
        attack=ATTACK,
        pct=PCT
    )

    # Global metrics (mean per-bin)
    adv_summary = summarize_per_bin_metrics(per_bin_df_adv)

    # Write CSV with two rows
    adv_csv_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv"
    df1 = pd.DataFrame([adv_metrics])
    df2 = pd.DataFrame([adv_summary])
    pd.concat([df1, df2], ignore_index=True).to_csv(adv_csv_path, index=False)

    print("Adversarial training metrics saved.\n")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("RESEARCH-GRADE CB-Focal MLP pipeline run completed.")
