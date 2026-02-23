import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

RESULTS_FOLDER = f"{SEG_FOLDER}/models_reducedscope"
SUMMARY_FOLDER = f"{RESULTS_FOLDER}/summary"
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

MODELS = ["LogReg", "XGBoost", "CatBoost", "MLP"]
PCTS = ["0_5", "1", "5", "10", "20"]

# ============================================================
# Helper: Load metrics CSV safely
# ============================================================
def load_metrics(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).iloc[0].to_dict()


# ============================================================
# MAIN PROCESSING
# ============================================================
rows = []

for model in MODELS:
    # -----------------------------
    # Load CLEAN baseline
    # -----------------------------
    clean_path = f"{RESULTS_FOLDER}/reducedscope_{model}_seg{SEGMENT_ID}_clean_baseline_metrics.csv"
    clean_metrics = load_metrics(clean_path)

    if clean_metrics is None:
        print(f"Missing baseline for {model}, skipping.")
        continue

    clean_acc = clean_metrics["accuracy"]

    # -----------------------------
    # Process each attack intensity
    # -----------------------------
    for pct in PCTS:

        # =========================
        # LABEL FLIP
        # =========================
        robust_path = f"{RESULTS_FOLDER}/reducedscope_{model}_seg{SEGMENT_ID}_robust_labelFlip_{pct}_metrics.csv"
        advtrain_path = f"{RESULTS_FOLDER}/reducedscope_{model}_seg{SEGMENT_ID}_advtrain_labelFlip_{pct}_metrics.csv"

        robust = load_metrics(robust_path)
        advtrain = load_metrics(advtrain_path)

        if robust:
            ASR = clean_acc - robust["accuracy"]
        else:
            ASR = None

        if robust and advtrain:
            ATG = advtrain["accuracy"] - robust["accuracy"]
        else:
            ATG = None

        rows.append({
            "model": model,
            "attack_type": "labelFlip",
            "pct": pct,
            "clean_acc": clean_acc,
            "robust_acc": robust["accuracy"] if robust else None,
            "advtrain_acc": advtrain["accuracy"] if advtrain else None,
            "ASR": ASR,
            "ATG": ATG
        })

        # =========================
        # SENSORY +1 dBm
        # =========================
        robust_path = f"{RESULTS_FOLDER}/reducedscope_{model}_seg{SEGMENT_ID}_robust_sensory_add1dBm_{pct}_metrics.csv"
        advtrain_path = f"{RESULTS_FOLDER}/reducedscope_{model}_seg{SEGMENT_ID}_advtrain_sensory_add1dBm_{pct}_metrics.csv"

        robust = load_metrics(robust_path)
        advtrain = load_metrics(advtrain_path)

        if robust:
            ASR = clean_acc - robust["accuracy"]
        else:
            ASR = None

        if robust and advtrain:
            ATG = advtrain["accuracy"] - robust["accuracy"]
        else:
            ATG = None

        rows.append({
            "model": model,
            "attack_type": "sensory_add1dBm",
            "pct": pct,
            "clean_acc": clean_acc,
            "robust_acc": robust["accuracy"] if robust else None,
            "advtrain_acc": advtrain["accuracy"] if advtrain else None,
            "ASR": ASR,
            "ATG": ATG
        })


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================
summary_df = pd.DataFrame(rows)
summary_path = f"{SUMMARY_FOLDER}/reducedscope_summary_seg{SEGMENT_ID}.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved summary table → {summary_path}")


# ============================================================
# OPTIONAL: Plot ASR curves for each model
# ============================================================
for model in MODELS:
    df_m = summary_df[summary_df["model"] == model]

    plt.figure(figsize=(10, 6))
    for attack in ["labelFlip", "sensory_add1dBm"]:
        df_a = df_m[df_m["attack_type"] == attack]
        plt.plot(df_a["pct"], df_a["ASR"], marker="o", label=attack)

    plt.title(f"Attack Success Rate (ASR) — {model} (Segment {SEGMENT_ID})")
    plt.xlabel("Poisoning Percentage (%)")
    plt.ylabel("ASR (Accuracy Drop)")
    plt.legend()
    plt.grid(True)

    plot_path = f"{SUMMARY_FOLDER}/ASR_curve_{model}_seg{SEGMENT_ID}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved ASR plot → {plot_path}")

print("\nPost‑processing complete.")
