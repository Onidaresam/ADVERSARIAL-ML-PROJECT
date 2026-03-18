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
OUT_BASE = f"{SEG_FOLDER}/feature_attacks"
OUT_RAND = f"{OUT_BASE}/random_noise"
OUT_BAND = f"{OUT_BASE}/band_noise"
OUT_MASK = f"{OUT_BASE}/masking"
OUT_FREQ = f"{OUT_BASE}/freq_specific"

os.makedirs(OUT_RAND, exist_ok=True)
os.makedirs(OUT_BAND, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)
os.makedirs(OUT_FREQ, exist_ok=True)

# Poisoning percentages
POISON_PCTS = [0.005, 0.01, 0.05, 0.10, 0.20]
MAGNITUDES = [0.5, 1.0]          # for random noise
BAND_WIDTH = 32                  # for structured band noise
BAND_MAG = 1.0                   # band noise magnitude
FREQ_MAG = 1.0                   # freq-specific distortion magnitude


# ============================================================
# 1. RANDOM FEATURE NOISE (±magnitude)
# ============================================================
def random_feature_noise(features, poison_percentage, magnitude):
    feats = features.copy()
    n_rows, n_cols = feats.shape
    total_elements = n_rows * n_cols

    k = int(poison_percentage * total_elements)
    flat_indices = np.random.choice(total_elements, size=k, replace=False)

    row_idx = flat_indices // n_cols
    col_idx = flat_indices % n_cols

    signs = np.random.choice([-1.0, 1.0], size=k)
    deltas = signs * magnitude

    feats[row_idx, col_idx] += deltas
    return feats


def generate_random_noise(features):
    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        for m in MAGNITUDES:
            m_str = str(m).replace('.', '_')

            pert = random_feature_noise(features, p, m)
            out_name = f"{OUT_RAND}/Features_RandNoise_seg{SEGMENT_ID}_{pct_str}pct_{m_str}dBm.csv"

            pd.DataFrame(pert).to_csv(out_name, index=False)
            print(f"[Random Noise] {p*100}% ±{m} dB → {out_name}")


# ============================================================
# 2. STRUCTURED BAND NOISE (band of width 32)
# ============================================================
def structured_band_noise(features, poison_percentage, band_width, magnitude):
    feats = features.copy()
    n_rows, n_cols = feats.shape

    start_col = np.random.randint(0, n_cols - band_width + 1)
    end_col = start_col + band_width

    k_rows = int(poison_percentage * n_rows)
    row_idx = np.random.choice(n_rows, size=k_rows, replace=False)

    feats[np.ix_(row_idx, np.arange(start_col, end_col))] += magnitude
    return feats, start_col, end_col


def generate_band_noise(features):
    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        pert, s, e = structured_band_noise(features, p, BAND_WIDTH, BAND_MAG)
        out_name = f"{OUT_BAND}/Features_BandNoise_seg{SEGMENT_ID}_{pct_str}pct_band{BAND_WIDTH}_plus{BAND_MAG}dBm.csv"

        pd.DataFrame(pert).to_csv(out_name, index=False)
        print(f"[Band Noise] {p*100}% band {s}–{e-1} → {out_name}")


# ============================================================
# 3. MASKING / ZEROING
# ============================================================
def masking_zero(features, poison_percentage):
    feats = features.copy()
    n_rows, n_cols = feats.shape
    total_elements = n_rows * n_cols

    k = int(poison_percentage * total_elements)
    flat_indices = np.random.choice(total_elements, size=k, replace=False)

    row_idx = flat_indices // n_cols
    col_idx = flat_indices % n_cols

    feats[row_idx, col_idx] = 0.0
    return feats


def generate_masking(features):
    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        pert = masking_zero(features, p)
        out_name = f"{OUT_MASK}/Features_MaskedZero_seg{SEGMENT_ID}_{pct_str}pct.csv"

        pd.DataFrame(pert).to_csv(out_name, index=False)
        print(f"[Masking] {p*100}% → {out_name}")


# ============================================================
# 4. FREQUENCY-SPECIFIC DISTORTION (label-guided)
# ============================================================
def freq_specific_distortion(features, labels, poison_percentage, magnitude):
    feats = features.copy()
    n_rows, n_cols = feats.shape

    ones_idx = np.argwhere(labels == 1)
    total_ones = ones_idx.shape[0]

    k = int(poison_percentage * total_ones)
    chosen = np.random.choice(total_ones, size=k, replace=False)

    sel_rows = ones_idx[chosen, 0]
    sel_cols = ones_idx[chosen, 1]

    signs = np.random.choice([-1.0, 1.0], size=k)
    deltas = signs * magnitude

    feats[sel_rows, sel_cols] += deltas
    return feats


def generate_freq_specific(features, labels):
    for p in POISON_PCTS:
        pct_str = str(p * 100).replace('.', '_')

        pert = freq_specific_distortion(features, labels, p, FREQ_MAG)
        out_name = f"{OUT_FREQ}/Features_FreqSpecific_seg{SEGMENT_ID}_{pct_str}pct_{FREQ_MAG}dBm.csv"

        pd.DataFrame(pert).to_csv(out_name, index=False)
        print(f"[Freq-Specific] {p*100}% → {out_name}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Running FEATURE attacks for segment {SEGMENT_ID} ({SEGMENT_TYPE})")

    # Load features (numeric columns only)
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_cols].astype(float).values

    # Load label matrix for freq-specific attack
    labels = pd.read_csv(label_file).values

    generate_random_noise(features)
    generate_band_noise(features)
    generate_masking(features)
    generate_freq_specific(features, labels)

    print("All FEATURE attacks completed.")
