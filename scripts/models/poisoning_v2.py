import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG — UPDATED FOR GOOGLE DRIVE
# ============================================================

SEGMENT_ID = "01"
SEGMENT_TYPE = "time"   # or "random"

# Path to your dataset folder in Google Drive
BASE = "/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/data/segments_time"

SEG_FOLDER = f"{BASE}/segment_{SEGMENT_ID}"

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

# NEW OUTPUT FOLDERS (SEPARATE FROM OLD EXPERIMENTS)
OUT_LABEL = f"{SEG_FOLDER}/label_flips_v2"
OUT_SENSORY = f"{SEG_FOLDER}/sensory_poison_v2"

os.makedirs(OUT_LABEL, exist_ok=True)
os.makedirs(OUT_SENSORY, exist_ok=True)

# Poisoning percentages
LABEL_PERCENTS = [0.5, 1, 5, 10, 20]        # percent of labels to flip
POISON_PCTS = [0.005, 0.01, 0.05, 0.10, 0.20]  # sensory poisoning percentages


# ============================================================
# 1. NEW LABEL FLIPPING (RANDOM BOTH DIRECTIONS)
# ============================================================
def flip_labels_random(df, percent):
    df = df.copy()
    n = len(df)
    k = int(n * (percent / 100))

    # Randomly choose rows to flip
    idx = np.random.choice(n, k, replace=False)

    # Randomly choose flip direction for each selected row
    random_flips = np.random.randint(0, 2, size=k)  # 0 or 1

    # Apply flipping: if random_flips[i] == 1 → flip, else leave unchanged
    df.iloc[idx] = np.abs(df.iloc[idx] - random_flips.reshape(-1, 1))

    return df


def generate_label_flips_v2():
    df_label = pd.read_csv(label_file)

    for p in LABEL_PERCENTS:
        flipped = flip_labels_random(df_label, p)
        out = f"{OUT_LABEL}/Label_Matrix_{SEGMENT_ID}_flip_v2_{str(p).replace('.', '_')}.csv"
        flipped.to_csv(out, index=False)
        print(f"[Label Flip v2] {p}% → {out}")


# ============================================================
# 2. NEW SENSORY POISONING (CONTINUOUS RANDOM VALUES)
# ============================================================
def poison_rssi_continuous(rssi, poison_pct):
    rssi = rssi.copy()
    n_rows, n_cols = rssi.shape
    total = n_rows * n_cols

    k = int(poison_pct * total)

    # Random positions
    flat_idx = np.random.choice(total, size=k, replace=False)
    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    # Continuous random values between -0.5 and +5
    deltas = np.random.uniform(-0.5, 5.0, size=k)

    rssi[row_idx, col_idx] += deltas
    return rssi


def generate_sensory_poison_v2():
    df = pd.read_csv(clean_file)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rssi = df[numeric_cols].astype(float).values

    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        poisoned = poison_rssi_continuous(rssi, p)
        out_name = f"{OUT_SENSORY}/RSSI_continuous_p{pct_str}.csv"
        pd.DataFrame(poisoned).to_csv(out_name, index=False)

        print(f"[Sensory v2] {p*100}% → {out_name}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Running NEW attacks for segment {SEGMENT_ID} ({SEGMENT_TYPE})")

    generate_label_flips_v2()
    generate_sensory_poison_v2()

    print("All NEW attacks completed.")
