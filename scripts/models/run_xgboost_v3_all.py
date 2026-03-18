# -*- coding: utf-8 -*-
"""
run_xgboost_v3_all.py

Row-matched, scientifically accurate sensory comparison:
- Split indices ONCE (deterministically) and reuse them for both clean and poisoned sets.
- Default sensory metrics are ENTRY-WISE (N×384); switch to row-wise via --sensory_mask_scope row.

Also:
- Uses Google Drive API + local cache (ensure_local_tree) for data paths
- Trains/predicts 384 XGBoost classifiers (one per bin)
- Computes baseline, robustness, and adversarial training metrics
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb

# ------------------------------------------------------------------
# Drive sync helper (import if available; fallback to local-only)
# ------------------------------------------------------------------
try:
    from scripts.gdrive_sync import ensure_local_tree
except ImportError:
    ensure_local_tree = None


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "adv_only"],
                    help="full = baseline + robustness + adversarial; adv_only = adversarial phase (baseline loaded or computed minimally if absent)")

parser.add_argument("--pct", type=str, required=True, help="poison percentage string, e.g., '1', '5', '0_5'")
parser.add_argument("--attack", type=str, required=True,
                    choices=["label_flip", "sensory_add1"])

parser.add_argument("--segment", type=str, default="01", help="segment id, e.g., '01'")

# Drive cache controls
parser.add_argument("--folder_id", type=str, default="1bg1W_xGKiP5z0zM1CBIo7u6KbOm-1U0O",
                    help="Google Drive folder ID for the project root")
parser.add_argument("--cache_root", type=str, default=r"C:\Data\project_cache",
                    help="Local cache root for synced Drive files")
parser.add_argument("--client_secret", type=str, default=r"C:\Dev\gpu\client_secret.json",
                    help="Path to client_secret.json")

# Sensory mask scope: entry-wise (N×384) or row-wise (N,)
parser.add_argument("--sensory_mask_scope",
                    choices=["entry", "row"],
                    default="entry",
                    help="How to define sensory attack mask. 'entry' = N×384 per-entry (default). 'row' = per-sample (any feature changed).")

# Deterministic split seed (so clean/poison indices match reproducibly)
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/test split (ensures clean & poisoned indices are matched)")

args = parser.parse_args()

SEGMENT_ID = args.segment
PCT = args.pct
ATTACK = args.attack
RANDOM_STATE = int(args.seed)


# ------------------------------------------------------------------
# Data roots: Drive cache (preferred) with local fallback
# ------------------------------------------------------------------
def get_base_path() -> str:
    # Preferred: build/refresh local cache from Drive folder
    if ensure_local_tree is not None:
        data_root = ensure_local_tree(folder_id=args.folder_id,
                                      cache_root=args.cache_root,
                                      client_secret_path=args.client_secret)
        # project_root/data/segments_time
        return str(Path(data_root) / "data" / "segments_time")
    # Fallback: repo-relative
    return str(Path(__file__).resolve().parents[2] / "data" / "segments_time")


BASE = get_base_path()
SEG_FOLDER = f"{BASE}/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips_v3"
SENSORY_FOLDER    = f"{SEG_FOLDER}/sensory_poison_v3"

OUT_FOLDER        = f"{SEG_FOLDER}/models_reducedscope/xgboost_v3"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_xgboost_v3"
DETAIL_FOLDER     = f"{SEG_FOLDER}/models_reducedscope/xgboost_v3_details"

Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(TEST_SPLIT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(DETAIL_FOLDER).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Device / GPU–CPU auto-fallback for XGBoost
# ------------------------------------------------------------------
def get_xgb_device_params():
    try:
        Xt = np.zeros((10, 1), dtype=np.float32)
        yt = np.zeros(10, dtype=np.int32)
        clf = xgb.XGBClassifier(
            n_estimators=1, max_depth=1, learning_rate=0.1,
            subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, reg_alpha=0.0,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False,
            tree_method="gpu_hist", predictor="gpu_predictor",
        )
        clf.fit(Xt, yt)
        params = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
        device_str = "GPU"
    except Exception:
        params = {"tree_method": "hist", "predictor": "cpu_predictor"}
        device_str = "CPU"
    return params, device_str


XGB_DEVICE_PARAMS, DEVICE_STR = get_xgb_device_params()
print(f"\n[DEVICE] XGBoost will run on: {DEVICE_STR}")
print(f"BASE: {BASE}\n")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_clean():
    df = pd.read_csv(clean_file)
    # Features sit in columns 1..384 (inclusive bounds in iloc slice)
    df_features = df.iloc[:, 1:385].astype(np.float32)
    X = df_features.values
    y = pd.read_csv(label_file).values  # shape (N, 384)
    return X, y


def load_poisoned(attack, pct):
    if attack == "label_flip":
        y_poison = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_v3_{pct}.csv"
        ).values  # (N, 384)
        X_clean, _ = load_clean()
        return X_clean, y_poison

    elif attack == "sensory_add1":
        sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p{pct}_v3.csv"
        dfp = pd.read_csv(sensory_file)
        dfp_features = dfp.iloc[:, 0:384].astype(np.float32)
        X_poison = dfp_features.values  # (N, 384 features)
        y_clean = pd.read_csv(label_file).values  # (N, 384 labels)
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
# ASR / ASenR (label_flip + sensory)
#   - label_flip uses entry-wise flipped mask (y_clean != y_poison) (N×384)
#   - sensory supports entry-wise (N×384)  or row-wise (N,) based on args.sensory_mask_scope
# ------------------------------------------------------------------
def compute_asr_label_flip(pred_poison, y_clean, y_poison):
    """
    ASR (label_flip): fraction of flipped entries where the model predicts the poisoned label.
    flipped_mask shape = (N, 384)
    """
    flipped_mask = (y_clean != y_poison)
    denom = flipped_mask.sum()
    if denom == 0:
        return 0.0, flipped_mask
    success = (pred_poison == y_poison) & flipped_mask
    return float(success.sum() / denom), flipped_mask


def _align_mask(mask, shape):
    """
    Accepts row-wise (N,) or entry-wise (N,B) mask and returns a mask of shape=shape.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        # Row-wise -> broadcast across bins
        m = m[:, None]
        if shape[1] > 1:
            m = np.broadcast_to(m, shape)
    elif m.ndim == 2:
        if m.shape != shape:
            raise ValueError(f"Mask shape {m.shape} != predictions shape {shape}; "
                             "use entry-wise mask with matching shape or row-wise mask.")
    else:
        raise ValueError("Mask must be 1-D (row-wise) or 2-D (entry-wise).")
    return m


def compute_asr_sensory(pred_clean, pred_poison, mask):
    """
    ASR (sensory): fraction of *poisoned entries* where the prediction changed vs. clean.
    Works with entry-wise (N×B) or row-wise (N,) mask.
    """
    pred_clean  = np.asarray(pred_clean)
    pred_poison = np.asarray(pred_poison)
    assert pred_clean.shape == pred_poison.shape, "predictions must have identical shape"
    m = _align_mask(mask, pred_clean.shape)
    changed = (pred_clean != pred_poison) & m
    denom = m.sum()
    return float(changed.sum() / denom) if denom else 0.0


def compute_asenr(pred_clean, pred_poison, mask):
    """
    ASenR: fraction of *clean (non-poisoned) entries* where the prediction changed.
    Works with entry-wise (N×B) or row-wise (N,) mask.
    """
    pred_clean  = np.asarray(pred_clean)
    pred_poison = np.asarray(pred_poison)
    assert pred_clean.shape == pred_poison.shape, "predictions must have identical shape"
    m = _align_mask(mask, pred_clean.shape)
    clean_positions = ~m
    denom = clean_positions.sum()
    if denom == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & clean_positions
    return float(changed.sum() / denom)


# ------------------------------------------------------------------
# XGBoost model bank (384 tasks)
# ------------------------------------------------------------------
def make_xgb_classifier():
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    params.update(XGB_DEVICE_PARAMS)
    return xgb.XGBClassifier(**params)


def train_xgb_models(X_train, y_train):
    models = []
    for j in range(y_train.shape[1]):
        yj = y_train[:, j]
        unq = np.unique(yj)
        if len(unq) == 1:
            models.append(("constant", int(unq[0])))
        else:
            clf = make_xgb_classifier()
            clf.fit(X_train, yj)
            models.append(("model", clf))
    return models


def predict_xgb_models(models, X):
    N, B = X.shape[0], len(models)
    preds = np.zeros((N, B), dtype=int)
    for j, (kind, obj) in enumerate(models):
        if kind == "constant":
            preds[:, j] = obj
        else:
            proba = obj.predict_proba(X)[:, 1]
            preds[:, j] = (proba >= 0.5).astype(int)
    return preds


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n[train_xgboost_v3] Segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}")
    print(f"Device mode: {DEVICE_STR}")
    print(f"Sensory mask scope: {args.sensory_mask_scope}")
    print(f"Random seed (row-matched splits): {RANDOM_STATE}\n")

    total_start = time.time()

    # ---- 0) Load full clean + poisoned datasets (aligned by row order) ----
    X_clean, y_clean = load_clean()
    Xp, yp = load_poisoned(ATTACK, PCT)

    # Sanity: same row count
    assert Xp.shape[0] == X_clean.shape[0], "Poisoned and clean must have same N (row count)."
    assert y_clean.shape[0] == X_clean.shape[0], "Label rows must match feature rows."
    N = X_clean.shape[0]

    # ---- 1) Split indices ONCE and reuse (row-matched) --------------------
    all_idx = np.arange(N)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )

    # Clean split (baseline)
    X_train, X_test = X_clean[idx_train], X_clean[idx_test]
    y_train, y_test = y_clean[idx_train], y_clean[idx_test]

    # Poisoned split (row-matched to the same indices)
    Xp_train, Xp_test = Xp[idx_train], Xp[idx_test]
    yp_train, yp_test = yp[idx_train], yp[idx_test]

        # ---- Baseline (clean) -------------------------------------------------
    BASELINE_MODEL_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_models.pkl"
    BASELINE_PRED_PATH  = f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_pred.npy"
    ROBUST_METRICS_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{ATTACK}_{PCT}_metrics.csv"
    ADV_MODEL_PATH      = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{ATTACK}_{PCT}_models.pkl"

    base_models = None
    pred_clean  = None

    if args.mode == "full":
        print("\n[BASELINE] Training 384 XGBoost models on CLEAN data (row-matched splits)...")
        t0 = time.time()
        base_models = train_xgb_models(X_train, y_train)
        joblib.dump(base_models, BASELINE_MODEL_PATH)
        pred_clean = predict_xgb_models(base_models, X_test)
        np.save(BASELINE_PRED_PATH, pred_clean)

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(y_test, pred_clean)
        df_bins = compute_per_bin_metrics(y_test, pred_clean)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        t1 = time.time()
        baseline_runtime = t1 - t0
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
            f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_metrics.csv",
            index=False
        )
    else:
        print("\n[BASELINE] Skipped (adv_only mode). Loading existing baseline...")
        base_models = joblib.load(BASELINE_MODEL_PATH)
        pred_clean = np.load(BASELINE_PRED_PATH)

    # ---- Robustness (clean-trained models on poisoned test) ---------------
    if args.mode == "full":
        print(f"\n[ROBUSTNESS] Evaluating clean-trained XGBoost → {ATTACK} {PCT}%")
        robust_start = time.time()

        # We already have row-matched splits: Xp_test, yp_test
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_{ATTACK}_{PCT}_Xtest.npy",
            Xp_test
        )
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_{ATTACK}_{PCT}_ytest.npy",
            yp_test
        )

        pred_robust = predict_xgb_models(base_models, Xp_test)
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{ATTACK}_{PCT}_pred.npy",
            pred_robust
        )

        if ATTACK == "label_flip":
            asr, flipped_mask = compute_asr_label_flip(
                pred_robust, y_clean=y_test, y_poison=yp_test
            )
            asenr = compute_asenr(pred_clean, pred_robust, flipped_mask)
        else:
            # Sensory: build mask according to scope
            if args.sensory_mask_scope == "entry":
                poisoned_mask = (Xp_test != X_test)
            else:  # "row"
                poisoned_mask = np.any(Xp_test != X_test, axis=1)
            asr = compute_asr_sensory(pred_clean, pred_robust, poisoned_mask)
            asenr = compute_asenr(pred_clean, pred_robust, poisoned_mask)

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test, pred_robust)
        df_bins = compute_per_bin_metrics(yp_test, pred_robust)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{ATTACK}_{PCT}_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        robust_end = time.time()
        robust_runtime = robust_end - robust_start
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

    # ---- Adversarial training (train on poisoned, test on poisoned) -------
    if args.mode in ["full", "adv_only"]:
        print(f"\n[ADV TRAIN] Training XGBoost on {ATTACK} {PCT}% poisoned data...")
        adv_start = time.time()

        # We already have Xp_train, Xp_test, yp_train, yp_test from row-matched split
        adv_models = train_xgb_models(Xp_train, yp_train)
        joblib.dump(adv_models, ADV_MODEL_PATH)

        pred_adv = predict_xgb_models(adv_models, Xp_test)
        np.save(
            f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{ATTACK}_{PCT}_pred.npy",
            pred_adv
        )

        if ATTACK == "label_flip":
            asr_adv, flipped_mask_adv = compute_asr_label_flip(
                pred_adv, y_clean=yp_test, y_poison=yp_test
            )
            asenr_adv = compute_asenr(pred_clean, pred_adv, flipped_mask_adv)
        else:
            if args.sensory_mask_scope == "entry":
                poisoned_mask_adv = (Xp_test != X_test)
            else:  # "row"
                poisoned_mask_adv = np.any(Xp_test != X_test, axis=1)
            asr_adv = compute_asr_sensory(pred_clean, pred_adv, poisoned_mask_adv)
            asenr_adv = compute_asenr(pred_clean, pred_adv, poisoned_mask_adv)

        p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test, pred_adv)
        df_bins = compute_per_bin_metrics(yp_test, pred_adv)
        df_bins.to_csv(
            f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{ATTACK}_{PCT}_perbin_metrics.csv",
            index=False
        )

        mean_acc = df_bins["accuracy"].mean()
        mean_f1 = df_bins["f1"].mean()

        adv_end = time.time()
        adv_runtime = adv_end - adv_start
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
            f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{ATTACK}_{PCT}_metrics.csv",
            index=False
        )

    # ---- Total runtime ----------------------------------------------------
    total_end = time.time()
    total_runtime = total_end - total_start
    print(f"\n[TOTAL] train_xgboost_v3 run completed in {total_runtime:.2f} seconds.\n")
