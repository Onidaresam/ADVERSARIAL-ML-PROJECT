import pandas as pd
from pathlib import Path

# -----------------------------------------
# FIXED ABSOLUTE PATHS (MATCH YOUR SETUP)
# -----------------------------------------
BASE = Path("/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/data/segments_time/segment_01/models_reducedscope")
MLP_DETAILS = BASE / "mlp_cb_focal_research_details"

POISON_PCTS = ["0_5", "1", "5", "10", "20"]
ATTACKS = ["label_flip", "sensory_add1"]

ACC_COL = "accuracy_full"
F1_COL = "f1_full"


# -----------------------------------------
# FUNCTION TO COMPUTE MEAN PER-BIN METRICS
# -----------------------------------------
def compute_mean_perbin(regime, attack, pct):
    """
    regime: clean_baseline, robust, advtrain
    attack: label_flip, sensory_add1
    pct:    0_5, 1, 5, 10, 20
    """

    # Baseline is computed once from a fixed file
    if regime == "clean_baseline":
        filename = "seg01_baseline_perbin_metrics.csv"
    else:
        filename = f"seg01_{regime}_{attack}_{pct}_perbin_metrics.csv"

    path = MLP_DETAILS / filename

    if not path.exists():
        print(f"❌ File not found: {path}")
        return None, None

    df = pd.read_csv(path)

    mean_acc = df[ACC_COL].mean()
    mean_f1 = df[F1_COL].mean()

    return mean_acc, mean_f1


# -----------------------------------------
# MAIN EXECUTION
# -----------------------------------------
def main():
    results = []

    print("\n======================")
    print("MLP BASELINE (computed once)")
    print("======================")

    # 1) BASELINE
    mean_acc, mean_f1 = compute_mean_perbin("clean_baseline", None, None)
    results.append({
        "regime": "clean_baseline",
        "attack": "none",
        "pct": "none",
        "mean_bin_accuracy": mean_acc,
        "mean_bin_f1": mean_f1
    })
    print(f"BASELINE → acc={mean_acc:.6f}, f1={mean_f1:.6f}\n")

    print("======================")
    print("MLP ROBUSTNESS + ADVTRAIN")
    print("======================")

    for attack in ATTACKS:
        print(f"\n--- ATTACK TYPE: {attack.upper()} ---")
        for pct in POISON_PCTS:
            print(f"\n  >>> Poisoning Level: {pct}")

            # ROBUSTNESS
            mean_acc, mean_f1 = compute_mean_perbin("robust", attack, pct)
            results.append({
                "regime": "robust",
                "attack": attack,
                "pct": pct,
                "mean_bin_accuracy": mean_acc,
                "mean_bin_f1": mean_f1
            })
            print(f"    ROBUST → acc={mean_acc:.6f}, f1={mean_f1:.6f}")

            # ADVERSARIAL TRAINING
            mean_acc, mean_f1 = compute_mean_perbin("advtrain", attack, pct)
            results.append({
                "regime": "advtrain",
                "attack": attack,
                "pct": pct,
                "mean_bin_accuracy": mean_acc,
                "mean_bin_f1": mean_f1
            })
            print(f"    ADVTRAIN → acc={mean_acc:.6f}, f1={mean_f1:.6f}")

    # Save results
    out_path = BASE / "mlp_mean_perbin_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("\n=======================================")
    print("ALL MLP MEAN PER-BIN METRICS COMPLETED")
    print("Saved to:", out_path)
    print("=======================================\n")


if __name__ == "__main__":
    main()
