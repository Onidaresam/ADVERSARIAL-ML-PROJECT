# -*- coding: utf-8 -*-
"""
train_catboost_v3_all.py

Row-matched, scientifically accurate sensory comparison:
- Split indices ONCE and reuse for clean/poisoned.
- Sensory mask supports entry-wise and row-wise.

GPU if available; otherwise CPU with multi-threading.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from catboost import CatBoostClassifier

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

# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=str, required=True)
parser.add_argument("--attack", type=str, required=True,
                    choices=["label_flip", "sensory_add1"])
parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "adv_only"])
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

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
def get_base_path():
    if ensure_local_tree is not None:
        data_root = ensure_local_tree(folder_id=args.folder_id,
                                      cache_root=args.cache_root,
                                      client_secret_path=args.client_secret)
        return str(Path(data_root) / "data" / "segments_time")
    return str(Path(__file__).resolve().parents[2] / "data" / "segments_time")

BASE = get_base_path()
SEG_FOLDER = f"{BASE}/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips_v3"
SENSORY_FOLDER    = f"{SEG_FOLDER}/sensory_poison_v3"

OUT_FOLDER        = f"{SEG_FOLDER}/models_reducedscope/catboost_v3"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_catboost_v3"
DETAIL_FOLDER     = f"{SEG_FOLDER}/models_reducedscope/catboost_v3_details"

Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(TEST_SPLIT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(DETAIL_FOLDER).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------
try:
    test_model = CatBoostClassifier(
        iterations=1,
        depth=1,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU"
    )
    test_model.fit([[0]], [0])
    DEVICE_STR = "GPU"
    TASK_TYPE = "GPU"
except Exception:
    DEVICE_STR = f"CPU ({os.cpu_count() or 8} threads)"
    TASK_TYPE = "CPU"

print(f"\n[train_catboost_v3_all] Segment {SEGMENT_ID}")
print(f"Attack: {ATTACK}, Pct: {PCT}")
print(f"Device: {DEVICE_STR}")
print(f"Sensory mask scope: {args.sensory_mask_scope}")
print(f"Random seed: {RANDOM_STATE}")
print(f"BASE: {BASE}\n")

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_clean():
    df = pd.read_csv(clean_file)
    df_features = df.iloc[:, 1:385].astype(np.float32)
    X = df_features.values
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
        sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p{pct}_v3.csv"
        dfp = pd.read_csv(sensory_file)
        dfp_features = dfp.iloc[:, 0:384].astype(np.float32)
        X_poison = dfp_features.values
        y_clean = pd.read_csv(label_file).values
        return X_poison, y_clean
    else:
        raise ValueError(f"Unsupported attack type: {attack}")

# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------
def compute_full_matrix_metrics(y_true, y_pred):
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy_full   = accuracy_score(y_true, y_pred)
    return precision_macro, recall_macro, f1_macro, accuracy_full

def compute_per_bin_metrics(y_true, y_pred):
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        rows.append([j, acc, prec, rec, f1])
    return pd.DataFrame(rows, columns=["bin_index", "accuracy", "precision", "recall", "f1"])

# ------------------------------------------------------------------
# ASR / ASenR
# ------------------------------------------------------------------
def compute_asr_label_flip(pred_poison, y_clean, y_poison):
    flipped_mask = (y_clean != y_poison)
    denom = flipped_mask.sum()
    if denom == 0:
        return 0.0, flipped_mask
    success = (pred_poison == y_poison) & flipped_mask
    return float(success.sum() / denom), flipped_mask

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

# ------------------------------------------------------------------
# CatBoost model bank
# ------------------------------------------------------------------
def train_cb_models(X_train, y_train):
    models = []
    for j in range(y_train.shape[1]):
        yj = y_train[:, j]
        unique_vals = np.unique(yj)
        if len(unique_vals) == 1:
            models.append(("constant", int(unique_vals[0])))
        else:
            clf = CatBoostClassifier(
                task_type=TASK_TYPE,
                loss_function="Logloss",
                depth=6,
                learning_rate=0.1,
                iterations=300,
                verbose=False,
                thread_count=os.cpu_count() or 8,
            )
            clf.fit(X_train, yj)
            models.append(("model", clf))
    return models

def predict_cb_models(models, X):
    N = X.shape[0]
    B = len(models)
    preds = np.zeros((N, B), dtype=int)
    for j, (kind, obj) in enumerate(models):
        if kind == "constant":
            preds[:, j] = obj
        else:
            preds[:, j] = obj.predict(X).astype(int)
    return preds

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    total_start = time.time()

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

    BASELINE_MODEL_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_v3_clean_models.pkl"
    BASELINE_PRED_PATH  = f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_v3_clean_pred.npy"
    ROBUST_METRICS_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_v3_robust_{ATTACK}_{PCT}_metrics.csv"
    ADV_MODEL_PATH      = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_v3_advtrain_{ATTACK}_{PCT}_models.pkl"

    base_models = None
    pred_clean  = None

    # BASELINE
    if args.mode == "full":
        print("\n[BASELINE] Training 384 CatBoost models on CLEAN data (row-matched)...")
        t0 = time.time()
        base_models = train_cb_models(X_train, y_train)
        joblib.dump(base_models, BASELINE_MODEL_PATH)

        pred_clean = predict_cb_models(base_models, X_test)
        np.save(BASELINE_PRED_PATH, pred_clean)

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(y_test, pred_clean)
        df_bins = compute_per_bin_metrics(y_test, pred_clean)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_v3_clean_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        baseline_runtime = time.time() - t0
        print(f"[BASELINE] Runtime: {baseline_runtime:.2f} seconds")

        df_main_clean = pd.DataFrame([{
            "phase": "clean",
            "attack": "none",
            "pct": "0",
            "ASR": 0.0,
            "ASenR": 0.0,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "accuracy_full": acc_full,
            "mean_perbin_accuracy": mean_acc,
            "mean_perbin_f1": mean_f1,
            "runtime_seconds": baseline_runtime,
            "device": DEVICE_STR
        }])
        df_main_clean.to_csv(
            f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_v3_clean_metrics.csv",
            index=False
        )
    else:
        print("[BASELINE] Recomputing predictions for alignment...")
        base_models = joblib.load(BASELINE_MODEL_PATH)
        pred_clean = predict_cb_models(base_models, X_test)

    # ROBUSTNESS
    if args.mode == "full":
        print(f"\n[ROBUSTNESS] Evaluating CatBoost → {ATTACK} {PCT}")
        robust_start = time.time()

        pred_robust = predict_cb_models(base_models, Xp_test)
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_v3_robust_{ATTACK}_{PCT}_pred.npy",
            pred_robust
        )

        if ATTACK == "label_flip":
            asr, flipped_mask = compute_asr_label_flip(
                pred_robust, y_clean=y_test, y_poison=yp_test
            )
            asenr = compute_asenr(pred_clean, pred_robust, flipped_mask)
        else:
            if args.sensory_mask_scope == "entry":
                poisoned_mask = (Xp_test != X_test)
            else:
                poisoned_mask = np.any(Xp_test != X_test, axis=1)
            asr = compute_asr_sensory(pred_clean, pred_robust, poisoned_mask)
            asenr = compute_asenr(pred_clean, pred_robust, poisoned_mask)

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test, pred_robust)
        df_bins = compute_per_bin_metrics(yp_test, pred_robust)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_v3_robust_{ATTACK}_{PCT}_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        robust_runtime = time.time() - robust_start
        print(f"[ROBUSTNESS] Runtime: {robust_runtime:.2f} seconds")

        df_main_robust = pd.DataFrame([{
            "phase": "robust",
            "attack": ATTACK,
            "pct": PCT,
            "ASR": asr,
            "ASenR": asenr,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "accuracy_full": acc_full,
            "mean_perbin_accuracy": mean_acc,
            "mean_perbin_f1": mean_f1,
            "runtime_seconds": robust_runtime,
            "device": DEVICE_STR
        }])
        df_main_robust.to_csv(ROBUST_METRICS_PATH, index=False)

    # ADV TRAIN
    if args.mode in ["full", "adv_only"]:
        print(f"\n[ADV TRAIN] Training CatBoost on {ATTACK} {PCT}% poisoned data...")
        adv_start = time.time()

        adv_models = train_cb_models(Xp_train, yp_train)
        joblib.dump(adv_models, ADV_MODEL_PATH)

        pred_adv = predict_cb_models(adv_models, Xp_test)
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_v3_advtrain_{ATTACK}_{PCT}_pred.npy",
            pred_adv
        )

        print("\n[DEBUG]")
        print("X_test:", X_test.shape)
        print("Xp_test:", Xp_test.shape)
        print("pred_clean:", pred_clean.shape)
        print("pred_adv:", pred_adv.shape)

        if ATTACK == "label_flip":
            asr_adv, flipped_mask_adv = compute_asr_label_flip(
                pred_adv, y_clean=yp_test, y_poison=yp_test
            )
            asenr_adv = compute_asenr(pred_clean, pred_adv, flipped_mask_adv)
        else:
            if args.sensory_mask_scope == "entry":
                poisoned_mask_adv = np.abs(Xp_test - X_test) > 1e-6
            else:
                poisoned_mask_adv = np.any(Xp_test != X_test, axis=1)
            asr_adv = compute_asr_sensory(pred_clean, pred_adv, poisoned_mask_adv)
            asenr_adv = compute_asenr(pred_clean, pred_adv, poisoned_mask_adv)

        assert pred_clean.shape == pred_adv.shape
        assert pred_clean.shape[0] == X_test.shape[0]

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test, pred_adv)
        df_bins = compute_per_bin_metrics(yp_test, pred_adv)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_v3_advtrain_{ATTACK}_{PCT}_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        adv_runtime = time.time() - adv_start
        print(f"[ADV TRAIN] Runtime: {adv_runtime:.2f} seconds")

        df_main_adv = pd.DataFrame([{
            "phase": "advtrain",
            "attack": ATTACK,
            "pct": PCT,
            "ASR": asr_adv,
            "ASenR": asenr_adv,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "accuracy_full": acc_full,
            "mean_perbin_accuracy": mean_acc,
            "mean_perbin_f1": mean_f1,
            "runtime_seconds": adv_runtime,
            "device": DEVICE_STR
        }])
        df_main_adv.to_csv(
            f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_v3_advtrain_{ATTACK}_{PCT}_metrics.csv",
            index=False
        )

    total_runtime = time.time() - total_start
    print(f"\n[TOTAL] train_catboost_v3_all run completed in {total_runtime:.2f} seconds.\n")
