import os
import subprocess
import datetime

# ============================================================
# CONFIG
# ============================================================

PCTS = ["0_5", "1", "5", "10", "20", "30", "50", "70"]
ATTACKS = ["label_flip", "sensory_add1"]

SCRIPT = "train_xgboost_v3.py"
SEGMENT_ID = "01"

# This is the REAL base for models on your Drive
BASE_MS = "/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/data/segments_time/segment_01/models_reducedscope"

OUT_FOLDER = f"{BASE_MS}/xgboost_v3"
DETAIL_FOLDER = f"{BASE_MS}/xgboost_v3_details"
LOG_FILE = "xgb_v3_smart_runner.log"


# ============================================================
# LOGGING
# ============================================================
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ============================================================
# FILE CHECK HELPERS
# ============================================================
def baseline_exists():
    model_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_models.pkl"
    metrics_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_metrics.csv"
    perbin_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_perbin_metrics.csv"
    return all(os.path.exists(p) for p in [model_path, metrics_path, perbin_path])


def robust_metrics_exists(attack, pct):
    metrics_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{attack}_{pct}_metrics.csv"
    perbin_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{attack}_{pct}_perbin_metrics.csv"
    return os.path.exists(metrics_path) and os.path.exists(perbin_path)


def advtrain_metrics_exists(attack, pct):
    metrics_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_metrics.csv"
    perbin_path = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_perbin_metrics.csv"
    model_path = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_models.pkl"
    return all(os.path.exists(p) for p in [metrics_path, perbin_path, model_path])


def file_corrupted(path):
    return os.path.exists(path) and os.path.getsize(path) == 0


# ============================================================
# RUN COMMAND
# ============================================================
def run(cmd):
    log(f"RUNNING: {' '.join(cmd)}")
    subprocess.run(cmd)


# ============================================================
# MAIN LOGIC
# ============================================================
def main():
    log("====================================================")
    log("SMART XGBOOST v3 RUNNER STARTED")
    log(f"OUT_FOLDER: {OUT_FOLDER}")
    log(f"DETAIL_FOLDER: {DETAIL_FOLDER}")
    log("====================================================")

    # 1. BASELINE
    if not baseline_exists():
        log("[BASELINE] Baseline not found. Running baseline once...")
        cmd = ["python3", SCRIPT, "--attack", "label_flip", "--pct", "0_5", "--mode", "full"]
        run(cmd)
    else:
        log("[BASELINE] Baseline already exists. Skipping baseline training.")

    # 2. POISONING LEVELS
    for attack in ATTACKS:
        for pct in PCTS:

            # Skip the baseline run itself
            if attack == "label_flip" and pct == "0_5":
                log(f"[SKIP] Baseline run ({attack}, {pct}) already completed.")
                continue

            robust_done = robust_metrics_exists(attack, pct)
            adv_done = advtrain_metrics_exists(attack, pct)

            robust_file = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{attack}_{pct}_perbin_metrics.csv"
            adv_file = f"{DETAIL_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_perbin_metrics.csv"

            if file_corrupted(robust_file):
                log(f"[WARNING] Corrupted robustness file for {attack} {pct}. Re-running robustness.")
                robust_done = False

            if file_corrupted(adv_file):
                log(f"[WARNING] Corrupted adversarial file for {attack} {pct}. Re-running adversarial training.")
                adv_done = False

            if robust_done and adv_done:
                log(f"[SKIP] All phases already completed for {attack} {pct}.")
                continue

            log(f"[RUN] Running {attack} {pct}% in adv_only mode...")
            cmd = ["python3", SCRIPT, "--attack", attack, "--pct", pct, "--mode", "adv_only"]
            run(cmd)

    log("====================================================")
    log("SMART XGBOOST v3 RUNNER COMPLETED")
    log("====================================================")


if __name__ == "__main__":
    main()
