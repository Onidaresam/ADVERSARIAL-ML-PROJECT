import pandas as pd
import os

BASE = "data"

def segment_time_based(clean_file, label_file, n_segments=12):

    df_clean = pd.read_csv(clean_file)
    df_label = pd.read_csv(label_file)

    total_rows = len(df_clean)
    segment_size = total_rows // n_segments

    print(f"Total rows: {total_rows}")
    print(f"Rows per segment: {segment_size}")

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else total_rows

        seg_clean = df_clean.iloc[start:end]
        seg_label = df_label.iloc[start:end]

        seg_folder = f"{BASE}/segments_time/segment_{i+1:02d}"
        os.makedirs(seg_folder, exist_ok=True)

        seg_clean.to_csv(f"{seg_folder}/Cleaned_Data_{i+1:02d}.csv", index=False)
        seg_label.to_csv(f"{seg_folder}/Label_Matrix_{i+1:02d}.csv", index=False)

        print(f"Segment {i+1:02d} saved: rows {start} to {end}")

if __name__ == "__main__":
    segment_time_based(
        clean_file="data/full_day/Cleaned_Data_full.csv",
        label_file="data/full_day/Label_Matrix_full.csv"
        #clean_file="ADVERSARIAL_ML_PROJECT/data/full_day/Cleaned_Data_full.csv",
        #label_file="ADVERSARIAL_ML_PROJECT/data/full_day/Label_Matrix_full.csv"
    )
