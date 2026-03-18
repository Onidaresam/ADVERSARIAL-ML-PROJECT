import os
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_research"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_logreg_research"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_research_details"

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

print("\n=== BIN INFO (ORIGINAL) ===")
print(f"Total bins (original): {NUM_BINS}")
print(f"Variable bins:         {len(variable_bins)}")
print(f"Constant-zero bins:    {len(constant_zero_bins)}")
print(f"Constant-one bins:     {len(constant_one_bins)}\n")


# ============================================================
# POISON FLAGS- ASR COMPUTATION
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
# DATA LOADING
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(np.float32).values
    y_full = pd.read_csv(label_file).values
    y_var = y_full[:, variable_bins]
    return X, y_full, y_var


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
        candidates = [
            f for f in os.listdir(SENSORY_FOLDER)
            if f.startswith(pattern) and f.endswith(".csv")
        ]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No sensory poisoning file found starting with: {pattern}"
            )
        sensory_file = candidates[0]
        X = pd.read_csv(os.path.join(SENSORY_FOLDER, sensory_file)).astype(np.float32).values
        y_full = pd.read_csv(label_file).values
        y_var = y_full[:, variable_bins]
        return X, y_full, y_var

    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

# ============================================================
# TRAIN 212 LOGISTIC REGRESSION MODELS (WITH PROBAS)
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
            solver="liblinear",
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            n_jobs=1,
        )
        clf.fit(X_train, yj)

        models.append(clf)
        constants.append(None)

    return models, constants


def predict_212_proba(models, constants, X):
    """
    Returns probabilities for the positive class for each bin.
    Shape: (n_samples, n_bins_var)
    """
    probs = []
    for clf, const in zip(models, constants):
        if const is not None:
            # Constant bin: probability is 0.0 or 1.0
            p = np.full(X.shape[0], float(const))
        else:
            p = clf.predict_proba(X)[:, 1]
        probs.append(p)
    return np.column_stack(probs)


def apply_thresholds(probs, thresholds, constants):
    """
    probs: (n_samples, n_bins_var)
    thresholds: (n_bins_var,)
    constants: list of None or 0/1
    """
    preds = []
    for j in range(len(variable_bins)):
        if constants[j] is not None:
            preds.append(np.full(probs.shape[0], constants[j], dtype=int))
        else:
            t = thresholds[j]
            preds.append((probs[:, j] >= t).astype(int))
    return np.column_stack(preds)

# ============================================================
# THRESHOLD TUNING
# ============================================================
def tune_thresholds(y_true, probs, constants):
    """
    For each bin, choose threshold in [0.05, 0.95] that maximizes F1.
    For constant bins, threshold is irrelevant; set to 0.5.
    """
    n_bins = probs.shape[1]
    thresholds = np.zeros(n_bins, dtype=float)

    grid = np.linspace(0.05, 0.95, 19)

    for j in range(n_bins):
        if constants[j] is not None:
            thresholds[j] = 0.5
            continue

        best_f1 = -1.0
        best_t = 0.5
        yt = y_true[:, j]
        pj = probs[:, j]

        for t in grid:
            yp = (pj >= t).astype(int)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[j] = best_t

    return thresholds

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
# PER-BIN METRICS + GLOBAL SUMMARY
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
    print(f"\nRunning Logistic Regression (RESEARCH MODE) for segment {SEGMENT_ID}")
    print(f"Attack type: {ATTACK}")
    print(f"Poisoning level: {PCT}%\n")

    global_start = time.time()

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    t0 = time.time()
    X_clean, y_clean_full, y_clean_var = load_clean_features_labels()
    print(f"[Time] Loaded clean data in {time.time() - t0:.2f}s")

    # Train/test split (CLEAN)
    X_train_full, X_test, y_train_full, y_test_full = train_test_split(
        X_clean, y_clean_full, test_size=0.2, shuffle=True
    )
    y_train_var_full = y_train_full[:, variable_bins]
    y_test_var = y_test_full[:, variable_bins]

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest_full.npy", y_test_full)

    # Train/val split (for threshold tuning)
    X_train, X_val, y_train_var, y_val_var = train_test_split(
        X_train_full, y_train_var_full, test_size=0.2, shuffle=True
    )

    # ========================================================
    # SCALER (CLEAN)
    # ========================================================
    scaler_clean = StandardScaler()
    X_train_scaled = scaler_clean.fit_transform(X_train)
    X_val_scaled = scaler_clean.transform(X_val)
    X_test_scaled = scaler_clean.transform(X_test)

    # ========================================================
    # BASELINE
    # ========================================================
    print("\n[BASELINE] Training Logistic Regression on CLEAN data...")
    t0 = time.time()
    models_clean, const_clean = train_212_models(X_train_scaled, y_train_var)
    print(f"[Time] Baseline training completed in {time.time() - t0:.2f}s")

    # Threshold tuning on validation set
    probs_val_clean = predict_212_proba(models_clean, const_clean, X_val_scaled)
    thresholds_clean = tune_thresholds(y_val_var, probs_val_clean, const_clean)

    # Evaluate on test set
    probs_test_clean = predict_212_proba(models_clean, const_clean, X_test_scaled)
    preds_var_clean = apply_thresholds(probs_test_clean, thresholds_clean, const_clean)
    full_pred_clean = reconstruct_full(preds_var_clean, y_test_full)

    # Save full predictions + labels for ASR/ASenR
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_clean_full_pred.npy", full_pred_clean)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_clean_ytest_full.npy", y_test_full)

    # Constant-bin poison flags (should be all zeros for clean, but keep for uniformity)
    poison_flags_clean = compute_constant_poison_flags(y_test_full)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_clean_poison_flags.npy", poison_flags_clean)


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

    per_bin_df_clean = compute_per_bin_metrics(
	y_test_var, 
	preds_var_clean, 
	f"baseline_{ATTACK}_{PCT}"
    )
    baseline_summary = summarize_per_bin_metrics(per_bin_df_clean)
    baseline_metrics.update(baseline_summary)

    baseline_filename = f"seg{SEGMENT_ID}_clean_baseline_{ATTACK}_{PCT}_metrics.csv"

    pd.DataFrame([baseline_metrics]).to_csv(
        f"{OUT_FOLDER}/{baseline_filename}",
        index=False
    )

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

    Xp_test_scaled = scaler_clean.transform(Xp_test)

    probs_robust = predict_212_proba(models_clean, const_clean, Xp_test_scaled)
    preds_var_robust = apply_thresholds(probs_robust, thresholds_clean, const_clean)
    full_pred_robust = reconstruct_full(preds_var_robust, yp_test_full)
    
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_robust_{ATTACK}_{PCT}_full_pred.npy", full_pred_robust)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_robust_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full)

    poison_flags_robust = compute_constant_poison_flags(yp_test_full)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_robust_{ATTACK}_{PCT}_poison_flags.npy", poison_flags_robust)


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

    per_bin_df_robust = compute_per_bin_metrics(
        yp_test_var, preds_var_robust, "robust", ATTACK, PCT
    )
    robust_summary = summarize_per_bin_metrics(per_bin_df_robust)
    robust_metrics.update(robust_summary)

    pd.DataFrame([robust_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING (POISONED TRAINING)
    # ========================================================
    print(f"\n[ADV TRAIN] Training LR on {ATTACK} {PCT}% poisoned data...")

    Xp_train_full, Xp_test_adv, yp_train_full, yp_test_full_adv = train_test_split(
        Xp, yp_full, test_size=0.2, shuffle=True
    )
    yp_train_var_full = yp_train_full[:, variable_bins]
    yp_test_var_adv = yp_test_full_adv[:, variable_bins]

    # Train/val split for poisoned training
    Xp_train, Xp_val, yp_train_var, yp_val_var = train_test_split(
        Xp_train_full, yp_train_var_full, test_size=0.2, shuffle=True
    )

    # Scaler for poisoned training
    scaler_adv = StandardScaler()
    Xp_train_scaled = scaler_adv.fit_transform(Xp_train)
    Xp_val_scaled = scaler_adv.transform(Xp_val)
    Xp_test_adv_scaled = scaler_adv.transform(Xp_test_adv)

    # Train models on poisoned data
    t0 = time.time()
    models_adv, const_adv = train_212_models(Xp_train_scaled, yp_train_var)
    print(f"[Time] Adversarial (poisoned) training completed in {time.time() - t0:.2f}s")

    # Threshold tuning on poisoned validation
    probs_val_adv = predict_212_proba(models_adv, const_adv, Xp_val_scaled)
    thresholds_adv = tune_thresholds(yp_val_var, probs_val_adv, const_adv)

    # Evaluate on poisoned test
    probs_adv = predict_212_proba(models_adv, const_adv, Xp_test_adv_scaled)
    preds_var_adv = apply_thresholds(probs_adv, thresholds_adv, const_adv)
    full_pred_adv = reconstruct_full(preds_var_adv, yp_test_full_adv)

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_advtrain_{ATTACK}_{PCT}_full_pred.npy", full_pred_adv)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_advtrain_{ATTACK}_{PCT}_ytest_full.npy", yp_test_full_adv)

    poison_flags_adv = compute_constant_poison_flags(yp_test_full_adv)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_advtrain_{ATTACK}_{PCT}_poison_flags.npy", poison_flags_adv)


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

    per_bin_df_adv = compute_per_bin_metrics(
        yp_test_var_adv, preds_var_adv, "advtrain", ATTACK, PCT
    )
    adv_summary = summarize_per_bin_metrics(per_bin_df_adv)
    adv_metrics.update(adv_summary)

    pd.DataFrame([adv_metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )

    print("Adversarial training metrics saved.\n")

    total_time = time.time() - global_start
    print(f"Total run time: {total_time/60:.2f} minutes")
    print("Logistic Regression (RESEARCH MODE) pipeline run completed.")
