import os
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib

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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_optimized"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_optimized"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_optimized_details"

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
# LOAD CLEAN DATA
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(np.float32).values
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
        pattern = f"RSSI_p{pct}"
        candidates = [f for f in os.listdir(SENSORY_FOLDER)
                      if f.startswith(pattern) and f.endswith(".csv")]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No sensory poisoning file found starting with: {pattern}")
        sensory_file = candidates[0]
        X = pd.read_csv(os.path.join(SENSORY_FOLDER, sensory_file)).astype(float).values
        y_full = pd.read_csv(label_file).values
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

# ============================================================
# TRAIN 212 LOGISTIC REGRESSION MODELS
# ============================================================
def train_212_models(X_train, y_train_var):
    models = []
    constants = []

    for j in range(len(variable_bins)):
        yj = y_train_var[:, j]
        unique_vals = np.unique(yj)

        # Constant bin in training split
        if len(unique_vals) == 1:
            models.append(None)
            constants.append(int(unique_vals[0]))
            continue


        clf = LogisticRegression(
            solver="saga",              # multi-core, faster for large data
            class_weight="balanced",
            max_iter=100,               # enough for convergence
            n_jobs=-1,                  # use all CPU cores
            warm_start=True,            # reuse previous solution
        )
        clf.fit(X_train, yj)



        models.append(clf)
        constants.append(None)

    return models, constants

# ============================================================
# PREDICT 212 MODELS
# ============================================================
def predict_212(models, constants, X):
    preds = []
    for clf, const in zip(models, constants):
        if const is not None:
            preds.append(np.full(X.shape[0], const))
        else:
            p = clf.predict_proba(X)[:, 1]
            preds.append((p >= 0.5).astype(int))
    return np.column_stack(preds)

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
# PER-BIN METRICS
# ============================================================
def compute_per_bin_metrics(y_true, y_pred, phase_name, attack=None, pct=None):
    rows_metrics = []
    rows_conf = []

    for j in range(len(variable_bins)):
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

    suffix = phase_name if attack is None else f"{phase_name}_{attack}_{pct}"

    pd.DataFrame(rows_metrics).to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_metrics.csv",
        index=False
    )
    pd.DataFrame(rows_conf).to_csv(
        f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_confusion.csv",
        index=False
    )

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning Logistic Regression (FAST MODE) for segment {SEGMENT_ID}")
    print(f"Attack type: {ATTACK}")
    print(f"Poisoning level: {PCT}%\n")

    global_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    t0 = time.time()
    X_clean, y_clean_full, y_clean_var = load_clean_features_labels()
    print(f"[Time] Loaded clean data in {time.time() - t0:.2f}s")

    # Train/test split
    X_train_full, X_test, y_train_full, y_test_full = train_test_split(
        X_clean, y_clean_full, test_size=0.2, shuffle=True
    )
    y_train_var_full = y_train_full[:, variable_bins]
    y_test_var = y_test_full[:, variable_bins]

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest_full.npy", y_test_full)

    # ========================================================
    # BASELINE
    # ========================================================
    print("\n[BASELINE] Training Logistic Regression on CLEAN data...")
    t0 = time.time()
    models_clean, const_clean = train_212_models(X_train_full, y_train_var_full)
    print(f"[Time] Baseline training completed in {time.time() - t0:.2f}s")

    preds_var_clean = predict_212(models_clean, const_clean, X_test)
    full_pred_clean = reconstruct_full(preds_var_clean, y_test_full)

    baseline_metrics = {
        "accuracy_var": accuracy_score(y_test_var, preds_var_clean),
        "precision_var": precision_score(y_test_var, preds_var_clean, average="macro", zero_division=0),
        "recall_var": recall_score(y_test_var, preds_var_clean, average="macro", zero_division=0),
        "f1_var": f1_score(y_test_var, preds_var_clean, average="macro", zero_division=0),

        "accuracy_full": accuracy_score(y_test_full, full_pred_clean),
        "precision_full": precision_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
        "recall_full": recall_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
        "f1_full": f1_score(y_test_full, full_pred_clean, average="macro", zero_division=0),
    }

    pd.DataFrame([baseline_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_metrics.csv",
        index=False
    )

    compute_per_bin_metrics(y_test_var, preds_var_clean, "baseline")

    print("Baseline metrics saved.")

    # ========================================================
    # ROBUSTNESS
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained LR → {ATTACK} {PCT}%")

    Xp, yp_full, yp_var = load_poisoned_features_labels(ATTACK, PCT)

    _, Xp_test, _, yp_test_full = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_test_var = yp_test_full[:, variable_bins]

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full)

    preds_var_robust = predict_212(models_clean, const_clean, Xp_test)
    full_pred_robust = reconstruct_full(preds_var_robust, yp_test_full)

    robust_metrics = {
        "accuracy_var": accuracy_score(yp_test_var, preds_var_robust),
        "precision_var": precision_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),
        "recall_var": recall_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),
        "f1_var": f1_score(yp_test_var, preds_var_robust, average="macro", zero_division=0),

        "accuracy_full": accuracy_score(yp_test_full, full_pred_robust),
        "precision_full": precision_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test_full, full_pred_robust, average="macro", zero_division=0),
    }

    pd.DataFrame([robust_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    compute_per_bin_metrics(yp_test_var, preds_var_robust, "robust", ATTACK, PCT)

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING
    # ========================================================
    print(f"\n[ADV TRAIN] Training LR on {ATTACK} {PCT}% poisoned data...")

    Xp_train_full, Xp_test_adv, yp_train_full, yp_test_full_adv = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_train_var_full = yp_train_full[:, variable_bins]
    yp_test_var_adv = yp_test_full_adv[:, variable_bins]

    models_adv, const_adv = train_212_models(Xp_train_full, yp_train_var_full)

    preds_var_adv = predict_212(models_adv, const_adv, Xp_test_adv)
    full_pred_adv = reconstruct_full(preds_var_adv, yp_test_full_adv)

    adv_metrics = {
        "accuracy_var": accuracy_score(yp_test_var_adv, preds_var_adv),
        "precision_var": precision_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "recall_var": recall_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "f1_var": f1_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),

        "accuracy_full": accuracy_score(yp_test_full_adv, full_pred_adv),
        "precision_full": precision_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
        "recall_full": recall_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
        "f1_full": f1_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0),
    }

    pd.DataFrame([adv_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    compute_per_bin_metrics(yp_test_var_adv, preds_var_adv, "advtrain", ATTACK, PCT)

    print("Adversarial training metrics saved.\n")

    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("Logistic Regression pipeline run completed.")
