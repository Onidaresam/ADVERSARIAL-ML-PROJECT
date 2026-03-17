import subprocess

# All poisoning percentages for v3
PCTS = ["0_5", "1", "5", "10", "20", "30", "50", "70"]

# Both attack types
ATTACKS = ["label_flip", "sensory_add1"]

SCRIPT = "train_xgboost_v3.py"

def run(cmd):
    print("\n====================================================")
    print("Running:", " ".join(cmd))
    print("====================================================\n")
    subprocess.run(cmd)

def main():
    for attack in ATTACKS:
        for pct in PCTS:
            cmd = ["python3", SCRIPT, "--attack", attack, "--pct", pct]
            run(cmd)

    print("\n\n==================== ALL XGBOOST v3 RUNS COMPLETED ====================\n")

if __name__ == "__main__":
    main()
