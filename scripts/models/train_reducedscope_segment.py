import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ============================================================
# CONFIG — CHANGE ONLY THESE TWO LINES
# ============================================================
SEGMENT_ID = "01"
SEGMENT_TYPE = "time"      # "time" or "random"
# ============================================================

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

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope"
os.makedirs(OUT_FOLDER, exist_ok=True)

# Poisoning percentages
PCTS = ["0_5", "1", "5", "10", "20"]


# ============================================================
# Utility: Load features (numeric only) + labels
# ============================================================
def load_clean_features_labels():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].astype(float).values
    y = pd.read_csv(label_file).values
    return X, y


def load_poisoned_features_labels(poison_type, pct):
    if poison_type == "label":
        label_path = f"{LABEL_FLIP_FOLDER}/Label_Matrix_{SEGMENT_ID}_flip_{pct}.csv"
        y = pd.read_csv(label_path).values
        X, _ = load_clean_features_labels()  # features unchanged
        return X, y

    elif poison_type == "sensory":
        feat_path = f"{SENSORY_FOLDER}/RSSI_p{pct}_add_1.csv"
        X = pd.read_csv(feat_path).astype(float).values
        y = pd.read_csv(label_file).values
        return X, y


# ============================================================
# Utility: Train/test split + scaling + metrics
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
    return metrics, preds, scaler


# ============================================================
# Models (wrapped in One‑Vs‑Rest for multi‑label)
# ============================================================
def get_models():
    return {
        "LogReg": OneVsRestClassifier(
            LogisticRegression(
                max_iter=200,
                solver="liblinear"
            )
        ),
        "XGBoost": OneVsRestClassifier(
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=-1
            )
        ),
        "CatBoost": OneVsRestClassifier(
            CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                verbose=False
            )
        ),
        "MLP": OneVsRestClassifier(
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300
            )
        )
    }


# ============================================================
# Save metrics + model + scaler
# ============================================================
def save_results(model_name, scenario, metrics, model, scaler):
    metrics_path = f"{OUT_FOLDER}/reducedscope_{model_name}_{scenario}_metrics.csv"
    model_path = f"{OUT_FOLDER}/reducedscope_{model_name}_{scenario}_model.pkl"
    scaler_path = f"{OUT_FOLDER}/reducedscope_{model_name}_{scenario}_scaler.pkl"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved metrics → {metrics_path}")
    print(f"Saved model   → {model_path}")
    print(f"Saved scaler  → {scaler_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print(f"Running REDUCED-SCOPE training for segment {SEGMENT_ID} ({SEGMENT_TYPE})")

    models = get_models()

    # ========================================================
    # PHASE 1 — BASELINE (clean → clean)
    # ========================================================
    X_clean, y_clean = load_clean_features_labels()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clean, y_clean, test_size=0.2, shuffle=True
    )

    for name, model in models.items():
        print(f"\n[BASELINE] Training {name} on CLEAN data...")
        metrics, _, scaler = evaluate_model(model, Xc_train, Xc_test, yc_train, yc_test)
        save_results(name, f"seg{SEGMENT_ID}_clean_baseline", metrics, model, scaler)

    # ========================================================
    # PHASE 2 — ROBUSTNESS (clean → poisoned)
    # ========================================================
    for pct in PCTS:
        # Label flips
        Xp, yp = load_poisoned_features_labels("label", pct)
        _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

        for name, model in models.items():
            print(f"\n[ROBUSTNESS] Testing {name} clean-trained → labelFlip {pct}%")
            metrics, _, scaler = evaluate_model(model, Xc_train, Xp_test, yc_train, yp_test)
            save_results(name, f"seg{SEGMENT_ID}_robust_labelFlip_{pct}", metrics, model, scaler)

        # Sensory +1 dBm
        Xp, yp = load_poisoned_features_labels("sensory", pct)
        _, Xp_test, _, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

        for name, model in models.items():
            print(f"\n[ROBUSTNESS] Testing {name} clean-trained → sensory +1dBm {pct}%")
            metrics, _, scaler = evaluate_model(model, Xc_train, Xp_test, yc_train, yp_test)
            save_results(name, f"seg{SEGMENT_ID}_robust_sensory_add1dBm_{pct}", metrics, model, scaler)

    # ========================================================
    # PHASE 3 — ADVERSARIAL TRAINING (poisoned → poisoned)
    # ========================================================
    for pct in PCTS:
        # Label flips
        Xp, yp = load_poisoned_features_labels("label", pct)
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

        for name, model in models.items():
            print(f"\n[ADV TRAIN] Training {name} on labelFlip {pct}%")
            metrics, _, scaler = evaluate_model(model, Xp_train, Xp_test, yp_train, yp_test)
            save_results(name, f"seg{SEGMENT_ID}_advtrain_labelFlip_{pct}", metrics, model, scaler)

        # Sensory +1 dBm
        Xp, yp = load_poisoned_features_labels("sensory", pct)
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, shuffle=True)

        for name, model in models.items():
            print(f"\n[ADV TRAIN] Training {name} on sensory +1dBm {pct}%")
            metrics, _, scaler = evaluate_model(model, Xp_train, Xp_test, yp_train, yp_test)
            save_results(name, f"seg{SEGMENT_ID}_advtrain_sensory_add1dBm_{pct}", metrics, model, scaler)

    print("\nAll REDUCED-SCOPE model training completed.")
