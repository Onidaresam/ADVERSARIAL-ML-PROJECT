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

BASE = "../../data/segments_time/segment_01"
OUT_FOLDER = f"{BASE}/models_reducedscope/xgboost_v3"
DETAIL_FOLDER = f"{BASE}/models_reducedscope/xgboost_v3_details"
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
    return os.path.exists(f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_clean_models.pkl")


def robust_metrics_exists(attack, pct):
    return os.path.exists(f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{attack}_{pct}_metrics.csv")


def advtrain_metrics_exists(attack, pct):
    return os.path.exists(f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_metrics.csv")


def file_corrupted(path):
    """A file is considered corrupted if it exists but is empty."""
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
    log("====================================================")

    # --------------------------------------------------------
    # 1. BASELINE (run once)
    # --------------------------------------------------------
    if not baseline_exists():
        log("[BASELINE] Baseline not found. Running baseline once...")
        cmd = ["python3", SCRIPT, "--attack", "label_flip", "--pct", "0_5", "--mode", "full"]
        run(cmd)
    else:
        log("[BASELINE] Baseline already exists. Skipping baseline training.")

    # --------------------------------------------------------
    # 2. POISONING LEVELS
    # --------------------------------------------------------
    for attack in ATTACKS:
        for pct in PCTS:

            # Skip the baseline run itself
            if attack == "label_flip" and pct == "0_5":
                log(f"[SKIP] Baseline run ({attack}, {pct}) already completed.")
                continue

            robust_file = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_robust_{attack}_{pct}_metrics.csv"
            adv_file = f"{OUT_FOLDER}/seg{SEGMENT_ID}_xgb_v3_advtrain_{attack}_{pct}_metrics.csv"

            robust_done = robust_metrics_exists(attack, pct)
            adv_done = advtrain_metrics_exists(attack, pct)

            # Corruption detection
            if file_corrupted(robust_file):
                log(f"[WARNING] Corrupted robustness file detected for {attack} {pct}. Re-running robustness.")
                robust_done = False

            if file_corrupted(adv_file):
                log(f"[WARNING] Corrupted adversarial file detected for {attack} {pct}. Re-running adversarial training.")
                adv_done = False

            # Skip if everything is done
            if robust_done and adv_done:
                log(f"[SKIP] All phases already completed for {attack} {pct}.")
                continue

            # Otherwise run in adv_only mode
            log(f"[RUN] Running {attack} {pct}% in adv_only mode...")
            cmd = ["python3", SCRIPT, "--attack", attack, "--pct", pct, "--mode", "adv_only"]
            run(cmd)

    log("====================================================")
    log("SMART XGBOOST v3 RUNNER COMPLETED")
    log("====================================================")


if __name__ == "__main__":
    main()
