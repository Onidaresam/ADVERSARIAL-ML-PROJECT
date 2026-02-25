import os
import numpy as np
import pandas as pd

SEGMENT_ID = "01"
BASE = "/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT"
SEG_FOLDER = f"{BASE}/data/segments_time/segment_{SEGMENT_ID}"
TEST_SPLIT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/testsplits_logreg_research"
BIN_INFO_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"

variable_bins = np.load(f"{BIN_INFO_FOLDER}/variable_bins.npy").astype(int)
constant_zero_bins = np.load(f"{BIN_INFO_FOLDER}/constant_zero_bins.npy").astype(int)
constant_one_bins = np.load(f"{BIN_INFO_FOLDER}/constant_one_bins.npy").astype(int)

def load_phase_arrays(model_name, phase, attack=None, pct=None):
    if phase == "clean":
        prefix = f"seg{SEGMENT_ID}_{model_name}_clean"
    else:
        prefix = f"seg{SEGMENT_ID}_{model_name}_{phase}_{attack}_{pct}"

    full_pred = np.load(f"{TEST_SPLIT_FOLDER}/{prefix}_full_pred.npy")
    y_full = np.load(f"{TEST_SPLIT_FOLDER}/{prefix}_ytest_full.npy")
    poison_flags = np.load(f"{TEST_SPLIT_FOLDER}/{prefix}_poison_flags.npy")
    return full_pred, y_full, poison_flags

def compute_asr_label_flip(y_clean, y_poison, pred_poison):
    # positions where label was flipped
    flipped = (y_clean != y_poison)
    if flipped.sum() == 0:
        return 0.0
    # success = model predicts attacker target (poisoned label)
    success = (pred_poison == y_poison) & flipped
    return success.sum() / flipped.sum()

def compute_asr_sensory(pred_clean, pred_poison, poisoned_mask):
    # poisoned_mask: where features were poisoned (we’ll start with constant bins via poison_flags)
    if poisoned_mask.sum() == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & poisoned_mask
    return changed.sum() / poisoned_mask.sum()

def compute_asenr(pred_clean, pred_poison, poisoned_mask):
    clean_positions = ~poisoned_mask
    if clean_positions.sum() == 0:
        return 0.0
    changed = (pred_clean != pred_poison) & clean_positions
    return changed.sum() / clean_positions.sum()

def main():
    rows = []

    model_name = "lr"  # later: loop over ["lr", "mlp", "xgb", "cat"]
    for attack in ["label_flip", "sensory_add1"]:
        for pct in ["0_5", "1", "5", "10", "20"]:
            # clean baseline
            pred_clean, y_clean, _ = load_phase_arrays(model_name, "clean")

            # robust (clean-trained → poisoned)
            pred_robust, y_poison, poison_flags = load_phase_arrays(
                model_name, "robust", attack, pct
            )

            # label_flip ASR
            if attack == "label_flip":
                asr = compute_asr_label_flip(y_clean, y_poison, pred_robust)
                poisoned_mask = (y_clean != y_poison)
            else:
                # sensory: start by using constant-bin poison flags as poisoned_mask
                poisoned_mask = poison_flags.astype(bool)
                asr = compute_asr_sensory(pred_clean, pred_robust, poisoned_mask)

            asenr = compute_asenr(pred_clean, pred_robust, poisoned_mask)

            rows.append({
                "model": model_name,
                "phase": "robust",
                "attack": attack,
                "pct": pct,
                "ASR": asr,
                "ASenR": asenr,
            })

    df = pd.DataFrame(rows)
    out_path = f"{SEG_FOLDER}/models_reducedscope/logreg_research/seg{SEGMENT_ID}_lr_asr_asenr.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
