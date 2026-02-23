import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG — CHANGE ONLY THESE TWO LINES TO SWITCH SEGMENTS
# ============================================================
SEGMENT_ID = "01"          # "01" → "12"
SEGMENT_TYPE = "time"      # "time" or "random"
# ============================================================

# Folder paths
BASE = "data"

SEG_FOLDER = (
    f"{BASE}/segments_time/segment_{SEGMENT_ID}"
    if SEGMENT_TYPE == "time"
    else f"{BASE}/segments_random/random_{SEGMENT_ID}"
)

clean_file = f"{SEG_FOLDER}/Cleaned_Data_{SEGMENT_ID}.csv"
label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

# Output folders
OUT_LABEL = f"{SEG_FOLDER}/label_flips"
OUT_SENSORY = f"{SEG_FOLDER}/sensory_poison"
OUT_MIXED = f"{SEG_FOLDER}/sensory_mixed"

os.makedirs(OUT_LABEL, exist_ok=True)
os.makedirs(OUT_SENSORY, exist_ok=True)
os.makedirs(OUT_MIXED, exist_ok=True)

# Poisoning percentages
LABEL_PERCENTS = [0.5, 1, 5, 10, 20]       # for label flipping
POISON_PCTS = [0.005, 0.01, 0.05, 0.10, 0.20]  # for sensory poisoning
MAGNITUDES = [0.5, 1.0]                    # dB shifts
MODS = np.array([0.5, 1.0, -0.5, -1.0])    # mixed poisoning deltas


# ============================================================
# 1. LABEL FLIPPING
# ============================================================
def flip_labels(df, percent):
    df = df.copy()
    n = len(df)
    k = int(n * (percent / 100))
    idx = np.random.choice(n, k, replace=False)
    df.iloc[idx] = 1 - df.iloc[idx]
    return df


def generate_label_flips():
    df_label = pd.read_csv(label_file)

    for p in LABEL_PERCENTS:
        flipped = flip_labels(df_label, p)
        out = f"{OUT_LABEL}/Label_Matrix_{SEGMENT_ID}_flip_{str(p).replace('.', '_')}.csv"
        flipped.to_csv(out, index=False)
        print(f"[Label Flip] {p}% → {out}")


# ============================================================
# 2. SENSORY POISONING (DETERMINISTIC ±0.5 / ±1 dB)
# ============================================================
def poison_rssi_matrix(rssi, poison_pct, magnitude, operation):
    rssi = rssi.copy()
    n_rows, n_cols = rssi.shape
    total = n_rows * n_cols

    k = int(poison_pct * total)

    flat_idx = np.random.choice(total, size=k, replace=False)
    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    if operation == "add":
        rssi[row_idx, col_idx] += magnitude
    else:
        rssi[row_idx, col_idx] -= magnitude

    return rssi


def generate_sensory_poison():
    df = pd.read_csv(clean_file)

    # Extract numeric RSSI columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rssi = df[numeric_cols].astype(float).values

    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        for m in MAGNITUDES:
            m_str = str(m).replace('.', '_')

            # +m
            poisoned_add = poison_rssi_matrix(rssi, p, m, "add")
            out_add = f"{OUT_SENSORY}/RSSI_p{pct_str}_add_{m_str}.csv"
            pd.DataFrame(poisoned_add).to_csv(out_add, index=False)
            print(f"[Sensory] +{m} dB at {p*100}% → {out_add}")

            # -m
            poisoned_sub = poison_rssi_matrix(rssi, p, m, "subtract")
            out_sub = f"{OUT_SENSORY}/RSSI_p{pct_str}_sub_{m_str}.csv"
            pd.DataFrame(poisoned_sub).to_csv(out_sub, index=False)
            print(f"[Sensory] -{m} dB at {p*100}% → {out_sub}")


# ============================================================
# 3. MIXED SENSORY POISONING (RANDOM ±0.5 / ±1 dB)
# ============================================================
def mixed_sensory_poisoning(rssi, poison_pct):
    rssi = rssi.copy()
    n_rows, n_cols = rssi.shape
    total = n_rows * n_cols

    k = int(poison_pct * total)

    flat_idx = np.random.choice(total, size=k, replace=False)
    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    deltas = np.random.choice(MODS, size=k)
    rssi[row_idx, col_idx] += deltas

    return rssi


def generate_mixed_sensory():
    df = pd.read_csv(clean_file)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rssi = df[numeric_cols].astype(float).values

    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        poisoned = mixed_sensory_poisoning(rssi, p)
        out_name = f"{OUT_MIXED}/RSSI_mixed_p{pct_str}.csv"
        pd.DataFrame(poisoned).to_csv(out_name, index=False)

        print(f"[Mixed Sensory] {p*100}% → {out_name}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Running attacks for segment {SEGMENT_ID} ({SEGMENT_TYPE})")

    generate_label_flips()
    generate_sensory_poison()
    generate_mixed_sensory()

    print("All attacks completed.")
