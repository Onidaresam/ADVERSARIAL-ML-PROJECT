# -*- coding: utf-8 -*-
"""
sensory_poison_v3structured.py

Structured bin-specific poisoning:
Each bin j receives a fixed offset in [-5, +5] dB.
For each poisoning percentage p, poison p% of entries in each bin independently.

One file per percentage.
No label-flip generation.
Fully compatible with --attack sensory_add1.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.gdrive_sync import ensure_local_tree

# ============================================================
# CONFIG
# ============================================================

SEGMENT_ID = "01"

# Resolve base path using ensure_local_tree()
BASE = ensure_local_tree(
    folder_id="1bg1W_xGKiP5z0zM1CBIo7u6KbOm-1U0O",
    cache_root=r"C:\Dev\gpu\cache",
    client_secret_path=r"C:\Dev\gpu\client_secret.json"
)

SEG_FOLDER = Path(BASE) / "data" / "segments_time" / f"segment_{SEGMENT_ID}"

clean_file = SEG_FOLDER / f"Cleaned_Data_{SEGMENT_ID}.csv"

OUT_SENSORY = SEG_FOLDER / "sensory_poison_v3structured"
OUT_SENSORY.mkdir(exist_ok=True)

# Poisoning percentages
LABEL_PERCENTS = [0.5, 1, 5, 10, 20, 30, 50, 70]
POISON_PCTS = [p / 100 for p in LABEL_PERCENTS]


# ============================================================
# Structured Bin Offsets
# ============================================================

def generate_bin_offsets(n_bins):
    np.random.seed(42)
    return np.random.uniform(-5.0, 5.0, size=n_bins)


# ============================================================
# Structured Poisoning
# ============================================================

def poison_rssi_structured(rssi, poison_frac, bin_offsets):
    rssi = rssi.copy()
    n_rows, n_cols = rssi.shape

    for j in range(n_cols):
        k = int(poison_frac * n_rows)
        idx = np.random.choice(n_rows, size=k, replace=False)
        rssi[idx, j] += bin_offsets[j]

    return rssi


def generate_sensory_poison_v3structured():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rssi = df[numeric_cols].astype(float).values

    n_bins = rssi.shape[1]
    bin_offsets = generate_bin_offsets(n_bins)

    print("Structured bin offsets (first 10 bins):", bin_offsets[:10])

    for p_frac, p_percent in zip(POISON_PCTS, LABEL_PERCENTS):
        pct_str = str(p_percent).replace('.', '_')

        poisoned = poison_rssi_structured(rssi, p_frac, bin_offsets)
        out_name = OUT_SENSORY / f"RSSI_structured_p{pct_str}_v3structured.csv"
        pd.DataFrame(poisoned).to_csv(out_name, index=False)

        print(f"[Structured v3] {p_percent}% → {out_name}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Running V3 structured sensory poisoning for segment {SEGMENT_ID}")
    generate_sensory_poison_v3structured()
    print("All structured V3 sensory attacks completed.")
