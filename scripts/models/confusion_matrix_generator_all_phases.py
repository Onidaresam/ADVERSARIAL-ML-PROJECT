import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import os

# ============================================================
# USER SETTINGS
# ============================================================
MODEL_NAME = "lr"   # "lr", "xgb", or "cb"
SEGMENT_ID = "01"

ATTACKS = ["label_flip", "sensory_add1"]
PCTS = ["0.5", "1", "5", "10", "20"]  # modify if needed

BASE = f"segments_time/segment_{SEGMENT_ID}/models_reducedscope"

MODEL_FOLDER = f"{BASE}/{MODEL_NAME}_v2"
DETAIL_FOLDER = f"{BASE}/{MODEL_NAME}_v2_details"
TEST_FOLDER = f"{BASE}/testsplits_{MODEL_NAME}_v2"

os.makedirs(DETAIL_FOLDER, exist_ok=True)

# ============================================================
# LOAD CLEAN TEST SPLIT
# ============================================================
X_test_clean = np.load(f"{TEST_FOLDER}/seg{SEGMENT_ID}_clean_Xtest.npy")
y_test_clean = np.load(f"{TEST_FOLDER}/seg{SEGMENT_ID}_clean_ytest.npy")

# ============================================================
# HELPER: COMPUTE AND SAVE CONFUSION MATRIX
# ============================================================
def save_confusion(y_true, y_pred, out_path):
    rows_conf = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows_conf.append({
            "bin_index": j,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        })
    df_conf = pd.DataFrame(rows_conf)
    df_conf.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# ============================================================
# 1. BASELINE CONFUSION MATRIX
# ============================================================
baseline_model_path = f"{MODEL_FOLDER}/seg{SEGMENT_ID}_{MODEL_NAME}_clean_model.pkl"
if os.path.exists(baseline_model_path):
    model = joblib.load(baseline_model_path)
    y_pred = model.predict(X_test_clean)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    out_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_baseline_perbin_confusion.csv"
    save_confusion(y_test_clean, y_pred, out_path)
else:
    print("Baseline model not found — skipping baseline.")

# ============================================================
# 2. ROBUSTNESS + ADVTRAIN CONFUSION MATRICES
# ============================================================
for attack in ATTACKS:
    for pct in PCTS:

        # ---------------------------
        # Load poisoned test split
        # ---------------------------
        poisoned_X_path = f"{TEST_FOLDER}/seg{SEGMENT_ID}_{attack}_{pct}_Xp_test.npy"
        poisoned_y_path = f"{TEST_FOLDER}/seg{SEGMENT_ID}_{attack}_{pct}_yp_test.npy"

        if not (os.path.exists(poisoned_X_path) and os.path.exists(poisoned_y_path)):
            print(f"Missing poisoned test split for {attack} {pct}% — skipping.")
            continue

        Xp_test = np.load(poisoned_X_path)
        yp_test = np.load(poisoned_y_path)

        # ---------------------------
        # ROBUSTNESS MODEL
        # ---------------------------
        robust_model_path = f"{MODEL_FOLDER}/seg{SEGMENT_ID}_{MODEL_NAME}_robust_{attack}_{pct}_model.pkl"
        if os.path.exists(robust_model_path):
            model = joblib.load(robust_model_path)
            y_pred = model.predict(Xp_test)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            out_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_robust_{attack}_{pct}_perbin_confusion.csv"
            save_confusion(yp_test, y_pred, out_path)
        else:
            print(f"No robust model for {attack} {pct}% — skipping.")

        # ---------------------------
        # ADVERSARIAL TRAINING MODEL
        # ---------------------------
        adv_model_path = f"{MODEL_FOLDER}/seg{SEGMENT_ID}_{MODEL_NAME}_advtrain_{attack}_{pct}_model.pkl"
        if os.path.exists(adv_model_path):
            model = joblib.load(adv_model_path)
            y_pred = model.predict(Xp_test)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            out_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_advtrain_{attack}_{pct}_perbin_confusion.csv"
            save_confusion(yp_test, y_pred, out_path)
        else:
            print(f"No advtrain model for {attack} {pct}% — skipping.")
