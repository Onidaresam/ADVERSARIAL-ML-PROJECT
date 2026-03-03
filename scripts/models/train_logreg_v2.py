import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_v2"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_logreg_v2"
DETAIL_FOLDER = f"{SEG_FOLDER}/models_reducedscope/logreg_v2_details"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)
os.makedirs(DETAIL_FOLDER, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def load_clean():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(float).values
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
        sensory_file = f"{SENSORY_FOLDER}/RSSI_continuous_p{pct}.csv"
        X_poison = pd.read_csv(sensory_file).astype(float).values
        y_clean = pd.read_csv(label_file).values
        return X_poison, y_clean

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
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning Logistic Regression v2 for segment {SEGMENT_ID}")
    print(f"Attack: {ATTACK}, Pct: {PCT}\n")

    # -----------------------------
    # LOAD CLEAN DATA
    # -----------------------------
    X_clean, y_clean = load_clean()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, shuffle=True
    )

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest.npy", y_test)

    # ========================================================
    # BASELINE (clean → clean)
    # ========================================================
    print("\n[BASELINE] Training Logistic Regression on CLEAN data...")

    base_clf = MultiOutputClassifier(
        LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            max_iter=1000
        )
    )

    base_clf.fit(X_train, y_train)
    pred_clean = base_clf.predict(X_test)

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_clean_pred.npy", pred_clean)

    # ========================================================
    # ROBUSTNESS (clean-trained → poisoned test)
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating clean-trained LR → {ATTACK} {PCT}%")

    Xp, yp = load_poisoned(ATTACK, PCT)
    _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest.npy", yp_test)

    pred_robust = base_clf.predict(Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_robust_{ATTACK}_{PCT}_pred.npy", pred_robust)

    # Compute ASR/ASenR
    if ATTACK == "label_flip":
        asr, flipped_mask = compute_asr_label_flip(pred_robust, y_clean=y_test, y_poison=yp_test)
        asenr = compute_asenr(pred_clean, pred_robust, flipped_mask)
    else:
        poisoned_mask = (Xp_test != X_test[:Xp_test.shape[0]])  # RSSI changed
        asr = compute_asr_sensory(pred_clean, pred_robust, poisoned_mask)
        asenr = compute_asenr(pred_clean, pred_robust, poisoned_mask)

    # Save metrics
    df = pd.DataFrame([{
        "phase": "robust",
        "attack": ATTACK,
        "pct": PCT,
        "ASR": asr,
        "ASenR": asenr
    }])

    df.to_csv(f"{OUT_FOLDER}/seg{SEGMENT_ID}_lr_robust_{ATTACK}_{PCT}_metrics.csv", index=False)

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    print(f"\n[ADV TRAIN] Training LR on {ATTACK} {PCT}% poisoned data...")

    Xp_train, Xp_test_adv, yp_train, yp_test_adv = train_test_split(
        Xp, yp, test_size=0.2, shuffle=True
    )

    adv_clf = MultiOutputClassifier(
        LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            max_iter=1000
        )
    )

    adv_clf.fit(Xp_train, yp_train)
    pred_adv = adv_clf.predict(Xp_test_adv)

    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_lr_advtrain_{ATTACK}_{PCT}_pred.npy", pred_adv)

    # Compute ASR/ASenR
    if ATTACK == "label_flip":
        asr_adv, flipped_mask_adv = compute_asr_label_flip(pred_adv, y_clean=yp_test_adv, y_poison=yp_test_adv)
        asenr_adv = compute_asenr(pred_clean, pred_adv, flipped_mask_adv)
    else:
        poisoned_mask_adv = (Xp_test_adv != X_test[:Xp_test_adv.shape[0]])
        asr_adv = compute_asr_sensory(pred_clean, pred_adv, poisoned_mask_adv)
        asenr_adv = compute_asenr(pred_clean, pred_adv, poisoned_mask_adv)

    df2 = pd.DataFrame([{
        "phase": "advtrain",
        "attack": ATTACK,
        "pct": PCT,
        "ASR": asr_adv,
        "ASenR": asenr_adv
    }])

    df2.to_csv(f"{OUT_FOLDER}/seg{SEGMENT_ID}_lr_advtrain_{ATTACK}_{PCT}_metrics.csv", index=False)

    print("\nLogistic Regression v2 pipeline completed.\n")
