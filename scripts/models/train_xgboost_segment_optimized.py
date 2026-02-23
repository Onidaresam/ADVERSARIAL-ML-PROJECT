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
from xgboost import XGBClassifier
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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/xgboost_optimized"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_optimized"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
METRICS_DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/xgboost_optimized_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(METRICS_DETAIL_FOLDER, exist_ok=True)

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
# HYPERPARAMETERS (FAST MODE ONLY)
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
# RECONSTRUCT FULL PREDICTIONS + POISONING DETECTION
# ============================================================
def reconstruct_full_and_detect(y_var_pred, y_full_true):
    N = y_full_true.shape[0]
    full_pred = np.zeros_like(y_full_true)

    full_pred[:, variable_bins] = y_var_pred

    if len(constant_zero_bins) > 0:
        full_pred[:, constant_zero_bins] = 0

    if len(constant_one_bins) > 0:
        full_pred[:, constant_one_bins] = 1

    poison_flags = np.zeros_like(y_full_true, dtype=int)

    if len(constant_zero_bins) > 0:
        cz_true = y_full_true[:, constant_zero_bins]
        poison_flags[:, constant_zero_bins] = (cz_true != 0).astype(int)

    if len(constant_one_bins) > 0:
        co_true = y_full_true[:, constant_one_bins]
        poison_flags[:, constant_one_bins] = (co_true != 1).astype(int)

    total_const_bins = len(constant_zero_bins) + len(constant_one_bins)
    total_const_positions = int(N * total_const_bins) if total_const_bins > 0 else 0
    total_poison_events = int(poison_flags.sum())

    poison_stats = {
        "total_constant_bins": int(total_const_bins),
        "total_constant_positions": total_const_positions,
        "total_poison_events": total_poison_events,
        "poison_rate_constant_positions": float(
            total_poison_events / total_const_positions
        ) if total_const_positions > 0 else 0.0,
    }

    return full_pred, poison_flags, poison_stats

# ============================================================
# SINGLE-MODEL TRAIN / PREDICT (MULTI-OUTPUT)
# ============================================================
def train_single_model(X_train, y_train_var):
    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        objective="binary:logistic",
    )

    clf.fit(X_train, y_train_var)
    return clf


def predict_single_model(model, X):
    proba = model.predict_proba(X)

    if isinstance(proba, list):
        probs = np.column_stack([p[:, 1] for p in proba])
    else:
        probs = proba

    preds = (probs >= 0.5).astype(int)
    return preds

# ============================================================
# PER-BIN METRICS AND CONFUSION SUMMARY
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
        f"{METRICS_DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_metrics.csv",
        index=False
    )
    conf_df.to_csv(
        f"{METRICS_DETAIL_FOLDER}/seg{SEGMENT_ID}_{suffix}_perbin_confusion.csv",
        index=False
    )

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning SINGLE-MODEL XGBoost (FAST MODE ONLY) for segment {SEGMENT_ID}")
    print(f"Attack type: {ATTACK}")
    print(f"Poisoning level: {PCT}%\n")

    global_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    t0 = time.time()
    X_clean, y_clean_full, y_clean_var = load_clean_features_labels()
    print(f"[Time] Loaded clean data in {time.time() - t0:.2f}s")

    # Train/test split on CLEAN data
    t0 = time.time()
    X_train_full, X_test, y_train_full, y_test_full = train_test_split(
        X_clean, y_clean_full, test_size=0.2, shuffle=True
    )
    y_train_var_full = y_train_full[:, variable_bins]
    y_test_var = y_test_full[:, variable_bins]

    # Train/val split preserved (A1)
    X_train, X_val, y_train_var, y_val_var = train_test_split(
        X_train_full, y_train_var_full, test_size=0.2, shuffle=True
    )
    print(f"[Time] Created train/val/test splits in {time.time() - t0:.2f}s")

    # Save clean test split
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest_full.npy", y_test_full)

    # ========================================================
    # BASELINE (clean → clean)
    # ========================================================
    print("\n[BASELINE] Training single XGBoost model on CLEAN data...")
    t0 = time.time()
    model_clean = train_single_model(X_train, y_train_var)
    print(f"[Time] Baseline training completed in {time.time() - t0:.2f}s")

    t0 = time.time()
    preds_var_clean = predict_single_model(model_clean, X_test)
    print(f"[Time] Baseline prediction completed in {time.time() - t0:.2f}s")

    full_pred_clean, poison_flags_clean, poison_stats_clean = reconstruct_full_and_detect(
        preds_var_clean, y_test_full
    )

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
        "poison_events_constant_bins": poison_stats_clean["total_poison_events"],
        "poison_rate_constant_positions": poison_stats_clean["poison_rate_constant_positions"],
    }

    pd.DataFrame([baseline_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_metrics.csv",
        index=False
    )
    joblib.dump(model_clean, f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_model.pkl")

    compute_per_bin_metrics(y_test_var, preds_var_clean, phase_name="baseline")

    print("Baseline metrics saved.")

    # ========================================================
    # ROBUSTNESS (clean-trained → poisoned test)
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained model → {ATTACK} {PCT}%")

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
    preds_var_robust = predict_single_model(model_clean, Xp_test)
    print(f"[Time] Robustness prediction completed in {time.time() - t0:.2f}s")

    full_pred_robust, poison_flags_robust, poison_stats_robust = reconstruct_full_and_detect(
        preds_var_robust, yp_test_full
    )

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
        "poison_events_constant_bins": poison_stats_robust["total_poison_events"],
        "poison_rate_constant_positions": poison_stats_robust["poison_rate_constant_positions"],
    }

    pd.DataFrame([robust_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump(model_clean, f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_model.pkl")

    compute_per_bin_metrics(yp_test_var, preds_var_robust, phase_name="robust", attack=ATTACK, pct=PCT)

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    print(f"\n[ADV TRAIN] Training model on {ATTACK} {PCT}% poisoned data...")

    t0 = time.time()
    Xp_train_full, Xp_test_adv, yp_train_full, yp_test_full_adv = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_train_var_full = yp_train_full[:, variable_bins]
    yp_test_var_adv = yp_test_full_adv[:, variable_bins]

    Xp_train, Xp_val, yp_train_var, yp_val_var = train_test_split(
        Xp_train_full, yp_train_var_full, test_size=0.2, shuffle=True
    )
    print(f"[Time] Created poisoned train/val/test splits in {time.time() - t0:.2f}s")

    t0 = time.time()
    model_adv = train_single_model(Xp_train, yp_train_var)
    print(f"[Time] Adversarial training completed in {time.time() - t0:.2f}s")

    t0 = time.time()
    preds_var_adv = predict_single_model(model_adv, Xp_test_adv)
    print(f"[Time] Adversarial prediction completed in {time.time() - t0:.2f}s")

    full_pred_adv, poison_flags_adv, poison_stats_adv = reconstruct_full_and_detect(
        preds_var_adv, yp_test_full_adv
    )

    metrics_var_adv = {
        "accuracy_var": accuracy_score(yp_test_var_adv, preds_var_adv),
        "precision_var": precision_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "recall_var": recall_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
        "f1_var": f1_score(yp_test_var_adv, preds_var_adv, average="macro", zero_division=0),
    }

    metrics_full_adv = {
        "accuracy_full": accuracy_score(yp_test_full_adv, full_pred_adv),
        "precision_full": precision_score(yp_test_full_adv, full_pred_adv, average="macro", zero_division=0)
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
        "poison_events_constant_bins": poison_stats_adv["total_poison_events"],
        "poison_rate_constant_positions": poison_stats_adv["poison_rate_constant_positions"],
    }

    pd.DataFrame([adv_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump(model_adv, f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_model.pkl")

    compute_per_bin_metrics(
        yp_test_var_adv,
        preds_var_adv,
        phase_name="advtrain",
        attack=ATTACK,
        pct=PCT
    )

    print("Adversarial training metrics saved.\n")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("SINGLE-MODEL XGBoost (FAST MODE, NO EARLY STOPPING) run completed.")
