import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
SEGMENT_ID = "01"
SEGMENT_TYPE = "time"

BASE = "data"

SEG_FOLDER = (
    f"{BASE}/segments_time/segment_{SEGMENT_ID}"
    if SEGMENT_TYPE == "time"
    else f"{BASE}/segments_random/random_{SEGMENT_ID}"
)

label_file = f"{SEG_FOLDER}/Label_Matrix_{SEGMENT_ID}.csv"

OUT_FOLDER = f"{SEG_FOLDER}/models_reducedscope/bin_info"
os.makedirs(OUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD LABEL MATRIX
# ============================================================
print(f"Loading label matrix from: {label_file}")
y = pd.read_csv(label_file).values  # shape: (N_samples, 384)

num_bins = y.shape[1]
print(f"Label matrix shape: {y.shape}")

# ============================================================
# IDENTIFY CONSTANT AND VARIABLE BINS
# ============================================================
constant_zero_bins = []
constant_one_bins = []
variable_bins = []

for i in range(num_bins):
    col = y[:, i]
    unique_vals = np.unique(col)

    if len(unique_vals) == 1:
        if unique_vals[0] == 0:
            constant_zero_bins.append(i)
        else:
            constant_one_bins.append(i)
    else:
        variable_bins.append(i)

print("\n=== SUMMARY ===")
print(f"Constant-zero bins: {len(constant_zero_bins)}")
print(f"Constant-one bins:  {len(constant_one_bins)}")
print(f"Variable bins:      {len(variable_bins)}")

# ============================================================
# SAVE RESULTS
# ============================================================
np.save(f"{OUT_FOLDER}/constant_zero_bins.npy", np.array(constant_zero_bins))
np.save(f"{OUT_FOLDER}/constant_one_bins.npy", np.array(constant_one_bins))
np.save(f"{OUT_FOLDER}/variable_bins.npy", np.array(variable_bins))

print("\nSaved:")
print(f" - constant_zero_bins.npy")
print(f" - constant_one_bins.npy")
print(f" - variable_bins.npy")

print("\nDone.")
