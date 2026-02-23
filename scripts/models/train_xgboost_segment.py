import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/xgboost"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEST_SPLIT_FOLDER, exist_ok=True)

# ============================================================
# LOAD CLEAN DATA
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(float).values
    y = pd.read_csv(label_file).values
    return X, y

# ============================================================
# LOAD POISONED DATA
# ============================================================
def load_poisoned_features_labels(attack_type, pct):
    if attack_type == "label_flip":
        y = pd.read_csv(f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_{pct}.csv").values
        X, _ = load_clean_features_labels()
        return X, y

    elif attack_type == "sensory_add1":
        X = pd.read_csv(f"{SENSORY_FOLDER}/RSSI_p{pct}_add_1.csv").astype(float).values
        y = pd.read_csv(label_file).values
        return X, y

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    return metrics, scaler

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\nRunning XGBoost for segment {SEGMENT_ID}")
    print(f"Attack type: {ATTACK}")
    print(f"Poisoning level: {PCT}%\n")

    # Load clean data
    X_clean, y_clean = load_clean_features_labels()

    # Train/test split (SAVED for post-processing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, shuffle=True
    )

    # Save clean test split
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy", X_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_clean_ytest.npy", y_test)

    # Define model
    model = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1
        )
    )

    # ========================================================
    # BASELINE (clean → clean)
    # ========================================================
    print("[BASELINE] Training XGBoost on CLEAN data...")
    metrics, scaler = evaluate_model(model, X_train, X_test, y_train, y_test)

    pd.DataFrame([metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_metrics.csv",
        index=False
    )
    joblib.dump(model, f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_model.pkl")
    joblib.dump(scaler, f"{OUT_FOLDER}/seg{SEGMENT_ID}_clean_baseline_scaler.pkl")

    print("Baseline metrics saved.")

    # ========================================================
    # ROBUSTNESS (clean → poisoned)
    # ========================================================
    print(f"\n[ROBUSTNESS] Evaluating XGBoost clean-trained → {ATTACK} {PCT}%")

    Xp, yp = load_poisoned_features_labels(ATTACK, PCT)
    _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

    # Save poisoned test split
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_Xtest.npy", Xp_test)
    np.save(f"{TEST_SPLIT_FOLDER}/seg{SEGMENT_ID}_{ATTACK}_{PCT}_ytest.npy", yp_test)

    metrics, scaler = evaluate_model(model, X_train, Xp_test, y_train, yp_test)

    pd.DataFrame([metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump(model, f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_model.pkl")
    joblib.dump(scaler, f"{OUT_FOLDER}/seg{SEGMENT_ID}_robust_{ATTACK}_{PCT}_scaler.pkl")

    print("Robustness metrics saved.")

    # ========================================================
    # ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    print(f"\n[ADV TRAIN] Training XGBoost on {ATTACK} {PCT}%")

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        Xp, yp, test_size=0.2, shuffle=True
    )

    metrics, scaler = evaluate_model(model, Xp_train, Xp_test, yp_train, yp_test)

    pd.DataFrame([metrics]).to_csv(
        f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_metrics.csv",
        index=False
    )
    joblib.dump(model, f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_model.pkl")
    joblib.dump(scaler, f"{OUT_FOLDER}/seg{SEGMENT_ID}_advtrain_{ATTACK}_{PCT}_scaler.pkl")

    print("Adversarial training metrics saved.\n")
    print("XGBoost run completed.")
