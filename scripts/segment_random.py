import pandas as pd
import numpy as np
import os

BASE = "data"

def segment_random(clean_file, label_file, n_segments=12):

    df_clean = pd.read_csv(clean_file)
    df_label = pd.read_csv(label_file)

    total_rows = len(df_clean)
    segment_size = total_rows // n_segments

    print(f"Total rows: {total_rows}")
    print(f"Rows per random segment: {segment_size}")

    # Shuffle indices for random segmentation
    shuffled_idx = np.random.permutation(total_rows)

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else total_rows

        idx = shuffled_idx[start:end]

        seg_clean = df_clean.iloc[idx]
        seg_label = df_label.iloc[idx]

        seg_folder = f"{BASE}/segments_random/random_{i+1:02d}"
        os.makedirs(seg_folder, exist_ok=True)

        seg_clean.to_csv(f"{seg_folder}/Cleaned_Data_random_{i+1:02d}.csv", index=False)
        seg_label.to_csv(f"{seg_folder}/Label_Matrix_random_{i+1:02d}.csv", index=False)

        print(f"Random segment {i+1:02d} saved: {len(idx)} rows")

if __name__ == "__main__":
    segment_random(
        clean_file="data/full_day/Cleaned_Data_full.csv",
        label_file="data/full_day/Label_Matrix_full.csv"
        #clean_file="ADVERSARIAL_ML_PROJECT/data/full_day/Cleaned_Data_full.csv",
        #label_file="ADVERSARIAL_ML_PROJECT/data/full_day/Label_Matrix_full.csv"
    )
