import os
import argparse
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import xgboost as xgb
import torch

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=str, required=True)
parser.add_argument("--attack", type=str, required=True, choices=["label_flip", "sensory_add1"])
args = parser.parse_args()

PCT = args.pct
ATTACK = args.attack

# ============================================================
# CONFIG
# ============================================================
SEGMENT_ID = "01"
BASE = "/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/ADVERSARIAL-ML-PROJECT/data/segments_time"
#BASE = "/kaggle/input/datasets/samuelonidare/adversarial-ml-project-data"
SEG_FOLDER = f"{BASE}/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips_v2"
SENSORY_FOLDER = f"{SEG_FOLDER}/sensory_poison_v2"

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/xgboost_v2"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_xgboost_v2"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/xgboost_v2_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)

# ============================================================
# DEVICE / GPU–CPU AUTO-FALLBACK
# ============================================================
def get_xgb_device_params():
    """
    Returns (params_dict, device_string) with automatic GPU→CPU fallback.
    """
    use_gpu = False
    try:
        if torch.cuda.is_available():
            use_gpu = True
    except Exception:
        use_gpu = False

    if use_gpu:
        params = {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor"
        }
        device_str = "GPU"
    else:
        params = {
            "tree_method": "hist",
            "predictor": "cpu_predictor"
        }
        device_str = "CPU"

    return params, device_str

XGB_DEVICE_PARAMS, DEVICE_STR = get_xgb_device_params()

print(f"\n[DEVICE] XGBoost will run on: {DEVICE_STR}\n")

# ============================================================
# DATA LOADING (MATCH LR/CATBOOST LOGIC)
# ============================================================
def load_clean():
    """
    Cleaned_Data_01.csv structure:
      col0   = timestamp (string)        -> DROP
      col1..col384 = RSSI features       -> KEEP (384 features)
      col385 = label (numeric)           -> DROP (labels come from Label_Matrix)
    """
    df = pd.read_csv(clean_file)
    df_features = df.iloc[:, 1:385].astype(np.float32)
    X = df_features.values
    y = pd.read_csv(label_file).values
    return X, y


def load_poisoned(attack, pct):
    if attack == "label_flip":
        y_poison = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_v2_{pct}.csv"
        ).values
        X_clean, _ = load_clean()
        return X_clean, y_poison

    elif attack == "sensory_add1":
        if pct == "0_5":
            sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p0_5.csv"
        else:
            sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p{pct}_0.csv"

        dfp = pd.read_csv(sensory_file)
        dfp_features = dfp.iloc[:, 0:384].astype(np.float32)
        X_poison = dfp_features.values
        y_clean = pd.read_csv(label_file).values
        return X_poison, y_clean

    else:
        raise ValueError(f"Unsupported attack type: {attack}")

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


def compute_asr_sensory(pred_clean, pred_poison, poisoned_mask):
    denom = poisoned_mask.sum()
    if denom == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & poisoned_mask
    return changed.sum() / denom


def compute_asenr(pred_clean, pred_poison, poisoned_mask):
    clean_positions = ~poisoned_mask
    denom = clean_positions.sum()
    if denom == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & clean_positions
    return changed.sum() / denom

# ============================================================
# TRAIN 384 XGBOOST MODELS WITH CONSTANT HANDLING
# ============================================================
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
    clf = xgb.XGBClassifier(**params)
    return clf


def train_xgb_models(X_train, y_train):
    models = []
    for j in range(y_train.shape[1]):
        yj = y_train[:, j]
        unique_vals = np.unique(yj)

        if len(unique_vals) == 1:
            models.append(("constant", int(unique_vals[0])))
        else:
            clf = make_xgb_classifier()
            clf.fit(X_train, yj)
            models.append(("model", clf))

    return models

# ============================================================
# PREDICT USING 384 XGBOOST MODELS
# ============================================================
def predict_xgb_models(models, X):
    N = X.shape[0]
    B = len(models)
    preds = np.zeros((N, B), dtype=int)

    for j, (kind, obj) in enumerate(models):
        if kind == "constant":
            preds[:, j] = obj
        else:
            proba = obj.predict_proba(X)[:, 1]
            preds[:, j] = (proba >= 0.5).astype(int)

    return preds

# ============================================================
# METRICS
# ============================================================
def compute_full_matrix_metrics(y_true, y_pred):
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy_full = accuracy_score(y_true, y_pred)
    return precision_macro, recall_macro, f1_macro, accuracy_full


def compute_per_bin_metrics(y_true, y_pred):
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        rows.append([j, acc, prec, rec, f1])

    df = pd.DataFrame(rows, columns=["bin_index", "accuracy", "precision", "recall", "f1"])
    return df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning XGBoost v2 for segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}")
    print(f"Device mode: {DEVICE_STR}\n")

    total_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    X_clean, y_clean = load_clean()

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, shuffle=True
    )

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest.npy", y_test)

    # ========================================================
    # BASELINE (clean → clean)
    # ========================================================
    print("\n[BASELINE] Training 384 XGBoost models on CLEAN data...")
    baseline_start = time.time()

    base_models = train_xgb_models(X_train, y_train)
    joblib.dump(base_models, f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_clean_models.pkl")

    pred_clean = predict_xgb_models(base_models, X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_clean_pred.npy", pred_clean)

    p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(y_test, pred_clean)
    df_bins = compute_per_bin_metrics(y_test, pred_clean)
    df_bins.to_csv(f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_clean_perbin_metrics.csv", index=False)

    mean_acc = df_bins["accuracy"].mean()
    mean_f1 = df_bins["f1"].mean()

    baseline_end = time.time()
    baseline_runtime = baseline_end - baseline_start
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

    df_main_clean.to_csv(f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_clean_metrics.csv", index=False)

    # ========================================================
    # ROBUSTNESS (clean-trained → poisoned test)
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained XGBoost → {ATTACK} {PCT}%")
    robust_start = time.time()

    Xp, yp = load_poisoned(ATTACK, PCT)
    _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest.npy", yp_test)

    pred_robust = predict_xgb_models(base_models, Xp_test)
    np.save(
        f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_robust_{ATTACK}_{PCT}_pred.npy",
        pred_robust
    )

    if ATTACK == "label_flip":
        asr, flipped_mask = compute_asr_label_flip(pred_robust, y_clean=y_test, y_poison=yp_test)
        asenr = compute_asenr(pred_clean, pred_robust, flipped_mask)
    else:
        poisoned_mask = (Xp_test != X_test[:Xp_test.shape[0]])
        asr = compute_asr_sensory(pred_clean, pred_robust, poisoned_mask)
        asenr = compute_asenr(pred_clean, pred_robust, poisoned_mask)

    p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test, pred_robust)
    df_bins = compute_per_bin_metrics(yp_test, pred_robust)
    df_bins.to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_robust_{ATTACK}_{PCT}_perbin_metrics.csv",
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

    df_main_robust.to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    print(f"\n[ADV TRAIN] Training XGBoost on {ATTACK} {PCT}% poisoned data...")
    adv_start = time.time()

    Xp_train, Xp_test_adv, yp_train, yp_test_adv = train_test_split(
        Xp, yp, test_size=0.2, shuffle=True
    )

    adv_models = train_xgb_models(Xp_train, yp_train)
    joblib.dump(
        adv_models,
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_advtrain_{ATTACK}_{PCT}_models.pkl"
    )

    pred_adv = predict_xgb_models(adv_models, Xp_test_adv)
    np.save(
        f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_xgb_advtrain_{ATTACK}_{PCT}_pred.npy",
        pred_adv
    )

    if ATTACK == "label_flip":
        asr_adv, flipped_mask_adv = compute_asr_label_flip(
            pred_adv, y_clean=yp_test_adv, y_poison=yp_test_adv
        )
        asenr_adv = compute_asenr(pred_clean, pred_adv, flipped_mask_adv)
    else:
        poisoned_mask_adv = (Xp_test_adv != X_test[:Xp_test_adv.shape[0]])
        asr_adv = compute_asr_sensory(pred_clean, pred_adv, poisoned_mask_adv)
        asenr_adv = compute_asenr(pred_clean, pred_adv, poisoned_mask_adv)

    p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(yp_test_adv, pred_adv)
    df_bins = compute_per_bin_metrics(yp_test_adv, pred_adv)
    df_bins.to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_advtrain_{ATTACK}_{PCT}_perbin_metrics.csv",
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
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    total_end = time.time()
    total_runtime = total_end - total_start
    print(f"\n[TOTAL] XGBoost v2 run completed in {total_runtime:.2f} seconds.\n")
