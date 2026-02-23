import os
import argparse
import time
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
from catboost import CatBoostClassifier
import joblib

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=str, required=True, help="Poisoning percentage: 0_5, 1, 5, 10, 20")
parser.add_argument("--attack", type=str, required=True, choices=["label_flip", "sensory_add1"])
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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/catboost_212"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_optimized"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/catboost_212_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)

# ============================================================
# LOAD BIN INFO
# ============================================================
variable_bins = np.load(f"{BIN_INFO_FOLDER}/variable_bins.npy").astype(int)
constant_zero_bins = np.load(f"{BIN_INFO_FOLDER}/constant_zero_bins.npy").astype(int)
constant_one_bins = np.load(f"{BIN_INFO_FOLDER}/constant_one_bins.npy").astype(int)

NUM_BINS = len(variable_bins) + len(constant_zero_bins) + len(constant_one_bins)

print("\n=== BIN INFO ===")
print(f"Total bins (original): {NUM_BINS}")
print(f"Variable bins:         {len(variable_bins)}")
print(f"Constant-zero bins:    {len(constant_zero_bins)}")
print(f"Constant-one bins:     {len(constant_one_bins)}\n")

# ============================================================
# HYPERPARAMETERS (FAST MODE)
# ============================================================
N_ESTIMATORS = 50
MAX_DEPTH = 2
LEARNING_RATE = 0.2

# ============================================================
# LOAD CLEAN DATA
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(float).values
    y_full = pd.read_csv(label_file).values
    y_var = y_full[:, variable_bins]
    return X, y_full, y_var

# ============================================================
# LOAD POISONED DATA
# ============================================================
def load_poisoned_features_labels(attack_type, pct):
    if attack_type == "label_flip":
        y_full = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_{pct}.csv"
        ).values
        X, _, _ = load_clean_features_labels()
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

    elif attack_type == "sensory_add1":
        X = pd.read_csv(
            f"{SENSORY_FOLDER}/RSSI_p{pct}_0_add_1_0.csv"
        ).astype(float).values
        y_full = pd.read_csv(label_file).values
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

# ============================================================
# RECONSTRUCT FULL PREDICTIONS
# ============================================================
def reconstruct_full(y_var_pred, y_full_true):
    N = y_full_true.shape[0]
    full_pred = np.zeros_like(y_full_true)

    full_pred[:, variable_bins] = y_var_pred

    if len(constant_zero_bins) > 0:
        full_pred[:, constant_zero_bins] = 0

    if len(constant_one_bins) > 0:
        full_pred[:, constant_one_bins] = 1

    return full_pred

# ============================================================
# TRAIN 212 MODELS WITH CONSTANT-BIN HANDLING
# ============================================================
def train_212_models(X_train, y_train_var):
    models = []
    constant_values = []

    for j in range(len(variable_bins)):
        yj = y_train_var[:, j]
        unique_vals = np.unique(yj)

        # If only one class present, store constant and skip training
        if len(unique_vals) == 1:
            models.append(None)
            constant_values.append(int(unique_vals[0]))
            continue

        clf = CatBoostClassifier(
            iterations=N_ESTIMATORS,
            depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            loss_function="Logloss",
            verbose=False,
            thread_count=-1,
        )
        clf.fit(X_train, yj)
        models.append(clf)
        constant_values.append(None)

    return models, constant_values

# ============================================================
# PREDICT WITH 212 MODELS + CONSTANTS
# ============================================================
def predict_212(models, constant_values, X):
    preds = []

    for clf, const in zip(models, constant_values):
        if const is not None:
            preds.append(np.full(X.shape[0], const, dtype=int))
        else:
            p = clf.predict_proba(X)[:, 1]
            preds.append((p >= 0.5).astype(int))

    return np.column_stack(preds)

# ============================================================
# PER-BIN METRICS
# ============================================================
def compute_per_bin_metrics(y_true, y_pred, phase_name, attack=None, pct=None):
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
            "bin_index": int(variable_bins[j]),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        rows_conf.append({
            "bin_index": int(variable_bins[j]),
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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning CatBoost 212-model pipeline for segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}\n")

    global_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    t0 = time.time()
    X_clean, y_clean_full, y_clean_var = load_clean_features_labels()
    print(f"[Time] Loaded clean data in {time.time() - t0:.2f}s")

    # Train/test split on CLEAN data
    X_train_full, X_test, y_train_full, y_test_full = train_test_split(
        X_clean, y_clean_full, test_size=0.2, shuffle=True
    )
    y_train_var_full = y_train_full[:, variable_bins]
    y_test_var = y_test_full[:, variable_bins]

    # Save clean test split (shared across models)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest_full.npy", y_test_full)

    # ========================================================
    # BASELINE (clean → clean)
    # ========================================================
    print("\n[BASELINE] Training 212 CatBoost models on CLEAN data...")
    t0 = time.time()
    models_clean, const_clean = train_212_models(X_train_full, y_train_var_full)
    print(f"[Time] Baseline training completed in {time.time() - t0:.2f}s")

    t0 = time.time()
    preds_var_clean = predict_212(models_clean, const_clean, X_test)
    print(f"[Time] Baseline prediction completed in {time.time() - t0:.2f}s")

    full_pred_clean = reconstruct_full(preds_var_clean, y_test_full)

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

    baseline_metrics = {
        **metrics_var_clean,
        **metrics_full_clean,
    }

    pd.DataFrame([baseline_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_metrics.csv",
        index=False
    )
    joblib.dump((models_clean, const_clean), f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_models.pkl")

    compute_per_bin_metrics(y_test_var, preds_var_clean, phase_name="baseline")

    print("Baseline metrics saved.")

    # ========================================================
    # ROBUSTNESS (clean-trained → poisoned test)
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained CatBoost → {ATTACK} {PCT}%")

    t0 = time.time()
    Xp, yp_full, yp_var = load_poisoned_features_labels(ATTACK, PCT)
    print(f"[Time] Loaded poisoned data in {time.time() - t0:.2f}s")

    t0 = time.time()
    _, Xp_test, _, yp_test_full = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_test_var = yp_test_full[:, variable_bins]
    print(f"[Time] Created poisoned test split in {time.time() - t0:.2f}s")

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full)

    t0 = time.time()
    preds_var_robust = predict_212(models_clean, const_clean, Xp_test)
    print(f"[Time] Robustness prediction completed in {time.time() - t0:.2f}s")

    full_pred_robust = reconstruct_full(preds_var_robust, yp_test_full)

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

    robust_metrics = {
        **metrics_var_robust,
        **metrics_full_robust,
    }

    pd.DataFrame([robust_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump((models_clean, const_clean), f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_models.pkl")

    compute_per_bin_metrics(yp_test_var, preds_var_robust, phase_name="robust", attack=ATTACK, pct=PCT)

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    print(f"\n[ADV TRAIN] Training 212 CatBoost models on {ATTACK} {PCT}% poisoned data...")

    t0 = time.time()
    Xp_train_full, Xp_test_adv, yp_train_full, yp_test_full_adv = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_train_var_full = yp_train_full[:, variable_bins]
    yp_test_var_adv = yp_test_full_adv[:, variable_bins]
    print(f"[Time] Created poisoned train/val/test splits in {time.time() - t0:.2f}s")

    t0 = time.time()
    models_adv, const_adv = train_212_models(Xp_train_full, yp_train_var_full)
    print(f"[Time] Adversarial training completed in {time.time() - t0:.2f}s")

    t0 = time.time()
    preds_var_adv = predict_212(models_adv, const_adv, Xp_test_adv)
    print(f"[Time] Adversarial prediction completed in {time.time() - t0:.2f}s")

    full_pred_adv = reconstruct_full(preds_var_adv, yp_test_full_adv)

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

    adv_metrics = {
        **metrics_var_adv,
        **metrics_full_adv,
    }

    pd.DataFrame([adv_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump((models_adv, const_adv), f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_models.pkl")

    compute_per_bin_metrics(yp_test_var_adv, preds_var_adv, phase_name="advtrain", attack=ATTACK, pct=PCT)

    print("Adversarial training metrics saved.\n")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("CatBoost 212-model pipeline run completed.")
