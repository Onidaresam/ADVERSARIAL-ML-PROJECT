# -*- coding: utf-8 -*-
"""
sensory_poison_v4.py

Generates sensory poisoning using 8 discrete strong noise values:
[-50, -40, -35, -30, +30, +35, +40, +50]

One poisoned file per percentage.
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

OUT_SENSORY = SEG_FOLDER / "sensory_poison_v4"
OUT_SENSORY.mkdir(exist_ok=True)

# Poisoning percentages
LABEL_PERCENTS = [0.5, 1, 5, 10, 20, 30, 50, 70]
POISON_PCTS = [p / 100 for p in LABEL_PERCENTS]

# Discrete strong noise values
NOISE_VALUES = np.array([-50, -40, -35, -30, 30, 35, 40, 50], dtype=float)


# ============================================================
# Strong Discrete Noise Poisoning
# ============================================================

def poison_rssi_discrete(rssi, poison_frac):
    rssi = rssi.copy()
    n_rows, n_cols = rssi.shape
    total = n_rows * n_cols

    k = int(poison_frac * total)

    flat_idx = np.random.choice(total, size=k, replace=False)
    row_idx = flat_idx // n_cols
    col_idx = flat_idx % n_cols

    deltas = np.random.choice(NOISE_VALUES, size=k, replace=True)
    rssi[row_idx, col_idx] += deltas

    return rssi


def generate_sensory_poison_v4():
    df = pd.read_csv(clean_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rssi = df[numeric_cols].astype(float).values

    for p_frac, p_percent in zip(POISON_PCTS, LABEL_PERCENTS):
        pct_str = str(p_percent).replace('.', '_')

        poisoned = poison_rssi_discrete(rssi, p_frac)
        out_name = OUT_SENSORY / f"RSSI_discrete_p{pct_str}_v4.csv"
        pd.DataFrame(poisoned).to_csv(out_name, index=False)

        print(f"[Sensory v4] {p_percent}% → {out_name}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Running V4 discrete-strong-noise sensory poisoning for segment {SEGMENT_ID}")
    generate_sensory_poison_v4()
    print("All V4 sensory attacks completed.")
