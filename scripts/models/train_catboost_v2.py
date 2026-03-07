import os
import argparse
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

from catboost import CatBoostClassifier

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
BASE = "/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/data/segments_time"

SEG_FOLDER = f"{BASE}/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

LABEL_FLIP_FOLDER = f"{SEG_FOLDER}/label_flips_v2"
SENSORY_FOLDER = f"{SEG_FOLDER}/sensory_poison_v2"

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/catboost_v2"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_catboost_v2"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/catboost_v2_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)

# ============================================================
# DATA LOADING (MATCH LR V2 LOGIC)
# ============================================================
def load_clean():
    """
    Cleaned_Data_01.csv structure (confirmed):
      col0   = timestamp (string)        -> DROP
      col1..col384 = RSSI features       -> KEEP (384 features)
      col385 = label (numeric)           -> DROP (labels come from Label_Matrix)
    """
    df = pd.read_csv(clean_file)

    # Keep only RSSI feature columns: 1..384 (iloc end is exclusive)
    df_features = df.iloc[:, 1:385].astype(np.float32)

    X = df_features.values
    y = pd.read_csv(label_file).values  # true labels

    return X, y


def load_poisoned(attack, pct):
    if attack == "label_flip":
        y_poison = pd.read_csv(
            f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_v2_{pct}.csv"
        ).values
        X_clean, _ = load_clean()
        return X_clean, y_poison

    elif attack == "sensory_add1":
        """
        Sensory poisoning CSV structure (confirmed):
          col0..col383 = poisoned RSSI features (384 numeric) -> KEEP
          col384       = extra numeric column                 -> DROP
        """
        if pct == "0_5":
            sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p0_5.csv"
        else:
            sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p{pct}_0.csv"

        dfp = pd.read_csv(sensory_file)

        # Keep only the first 384 columns as features
        dfp_features = dfp.iloc[:, 0:384].astype(np.float32)
        X_poison = dfp_features.values

        y_clean = pd.read_csv(label_file).values
        return X_poison, y_clean

    else:
        raise ValueError(f"Unsupported attack type: {attack}")

# ============================================================
# ASR / ASenR (SAME AS LR V2)
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
# TRAIN 384 CATBOOST MODELS WITH CONSTANT HANDLING
# ============================================================
def train_cb_models(X_train, y_train):
    models = []
    for j in range(y_train.shape[1]):
        yj = y_train[:, j]
        unique_vals = np.unique(yj)

        if len(unique_vals) == 1:
            models.append(("constant", int(unique_vals[0])))
        else:
            clf = CatBoostClassifier(
                task_type="CPU",
                #devices="0",
                loss_function="Logloss",
                depth=6,
                learning_rate=0.1,
                iterations=300,
                verbose=False,
                class_weights=None  # you can tune this if needed
            )
            clf.fit(X_train, yj)
            models.append(("model", clf))

    return models

# ============================================================
# PREDICT USING 384 CATBOOST MODELS
# ============================================================
def predict_cb_models(models, X):
    N = X.shape[0]
    B = len(models)
    preds = np.zeros((N, B), dtype=int)

    for j, (kind, obj) in enumerate(models):
        if kind == "constant":
            preds[:, j] = obj
        else:
            preds[:, j] = obj.predict(X)
            preds[:, j] = preds[:, j].astype(int)

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
    print(f"\nRunning CatBoost v2 for segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}\n")

    # -----------------------------
    # GPU DETECTION (REAL)
    # -----------------------------
    try:
        from catboost import CatBoostClassifier
        test_model = CatBoostClassifier(
            iterations=1,
            depth=1,
            learning_rate=0.1,
            loss_function="Logloss",
            task_type="GPU"
        )
        test_model.fit([[0]], [0])
        DEVICE_STR = "GPU"
    except Exception:
        DEVICE_STR = "CPU"

    print(f"[DEVICE] CatBoost will run on: {DEVICE_STR}")

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
    # CHECKPOINT PATHS
    # ========================================================
    BASELINE_MODEL_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_clean_models.pkl"
    BASELINE_PRED_PATH  = f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_clean_pred.npy"
    ROBUST_METRICS_PATH = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_robust_{ATTACK}_{PCT}_metrics.csv"
    ADV_MODEL_PATH      = f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_advtrain_{ATTACK}_{PCT}_models.pkl"

    # ========================================================
    # BASELINE (clean → clean) — RESTART-SAFE
    # ========================================================
    if args.mode == "full":
        if not (os.path.exists(BASELINE_MODEL_PATH) and os.path.exists(BASELINE_PRED_PATH)):
            print("\n[BASELINE] No baseline found → training 384 CatBoost models...")

            base_models = train_cb_models(X_train, y_train)
            joblib.dump(base_models, BASELINE_MODEL_PATH)

            pred_clean = predict_cb_models(base_models, X_test)
            np.save(BASELINE_PRED_PATH, pred_clean)

            p_macro, r_macro, f1_macro, acc_full = compute_full_matrix_metrics(y_test, pred_clean)
            df_bins = compute_per_bin_metrics(y_test, pred_clean)
            df_bins.to_csv(f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_clean_perbin_metrics.csv", index=False)

            mean_acc = df_bins["accuracy"].mean()
            mean_f1 = df_bins["f1"].mean()

            df_main = pd.DataFrame([{
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
                "device": DEVICE_STR
            }])

            df_main.to_csv(f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_clean_metrics.csv", index=False)

        else:
            print("\n[BASELINE] Found existing baseline → loading.")
            base_models = joblib.load(BASELINE_MODEL_PATH)
            pred_clean = np.load(BASELINE_PRED_PATH)

    # ========================================================
    # LOAD POISONED DATA ONCE
    # ========================================================
    if args.mode == "full":
        Xp, yp = load_poisoned(ATTACK, PCT)

    # ========================================================
    # ROBUSTNESS (clean-trained → poisoned test) — RESTART-SAFE
    # ========================================================
    if args.mode == "full":
        if not os.path.exists(ROBUST_METRICS_PATH):
            print(f"\n[ROBUSTNESS] No robustness results → evaluating CatBoost → {ATTACK} {PCT}%")

            _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

            pred_robust = predict_cb_models(base_models, Xp_test)
            np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_robust_{ATTACK}_{PCT}_pred.npy", pred_robust)

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
                f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_robust_{ATTACK}_{PCT}_perbin_metrics.csv",
                index=False
            )

            mean_acc = df_bins["accuracy"].mean()
            mean_f1 = df_bins["f1"].mean()

            df_main = pd.DataFrame([{
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
                "device": DEVICE_STR
            }])

            df_main.to_csv(ROBUST_METRICS_PATH, index=False)

        else:
            print(f"\n[ROBUSTNESS] Found existing robustness results → skipping.")

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned) — RESTART-SAFE
    # ========================================================
    if args.mode == "full" or args.mode == "adv_only":
        if not os.path.exists(ADV_MODEL_PATH):
            print(f"\n[ADV TRAIN] No adversarial model → training CatBoost on {ATTACK} {PCT}% poisoned data...")

            Xp_train, Xp_test_adv, yp_train, yp_test_adv = train_test_split(
                Xp, yp, test_size=0.2, shuffle=True
            )

            adv_models = train_cb_models(Xp_train, yp_train)
            joblib.dump(adv_models, ADV_MODEL_PATH)

            pred_adv = predict_cb_models(adv_models, Xp_test_adv)
            np.save(
                f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_cb_advtrain_{ATTACK}_{PCT}_pred.npy",
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
                f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_cb_advtrain_{ATTACK}_{PCT}_perbin_metrics.csv",
                index=False
            )

            mean_acc = df_bins["accuracy"].mean()
            mean_f1 = df_bins["f1"].mean()

            df_main = pd.DataFrame([{
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
                "device": DEVICE_STR
            }])

            df_main.to_csv(
                f"{OUT_FOLDER}/seg{SEGMENT_ID}_cb_advtrain_{ATTACK}_{PCT}_metrics.csv",
                index=False
            )

        else:
            print(f"\n[ADV TRAIN] Found existing adversarial model → skipping.")

    print("\nCatBoost v2 pipeline completed.\n")
