%%writefile scripts/models/merge_all_metrics_separately.py
import os
import pandas as pd

SEGMENT_ID = "01"
BASE = f"data/segments_time/segment_{SEGMENT_ID}/models_reducedscope"

MODEL_FOLDERS = {
    "lr":  "logreg_research",
    "mlp": "mlp_cb_focal_research",
    "xgb": "xgboost_optimized",
    "cat": "catboost_212",
}

METRIC_COLS = [
    "accuracy_var", "precision_var", "recall_var", "f1_var",
    "accuracy_full", "precision_full", "recall_full", "f1_full"
]

# Load mean bin accuracy for MLP only
mlp_mean_bin_accuracy = None
mlp_mean_file = os.path.join(BASE, "mlp_mean_perbin_results.csv")
if os.path.exists(mlp_mean_file):
    df_mlp = pd.read_csv(mlp_mean_file)
    mlp_mean_bin_accuracy = df_mlp["mean_accuracy"].iloc[0]


def parse_metric_filename(fname):
    """
    Expected filename format:
    seg01_<model>_<phase>_<attack>_<pct>_metrics.csv
    """
    parts = fname.replace(".csv", "").split("_")
    if len(parts) < 7:
        return None

    model = parts[1]
    phase = parts[2]

    if parts[3] == "label" and parts[4] == "flip":
        attack = "label_flip"
        pct = parts[5] if parts[5] != "0" else "0_5"
    elif parts[3] == "sensory" and parts[4] == "add1":
        attack = "sensory_add1"
        pct = parts[5] if parts[5] != "0" else "0_5"
    else:
        return None

    return model, phase, attack, pct


rows = []

for model, folder in MODEL_FOLDERS.items():
    folder_path = os.path.join(BASE, folder)

    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue

        parsed = parse_metric_filename(fname)
        if parsed is None:
            continue

        model_id, phase, attack, pct = parsed

        df = pd.read_csv(os.path.join(folder_path, fname))
        row = df.iloc[0].to_dict()

        # Special rule for LR: mean_perbin accuracy is inside the LR metrics file
        if model_id == "lr" and "mean_perbin_accuracy" in row:
            row["accuracy_var"] = row["mean_perbin_accuracy"]
            row["accuracy_full"] = row["mean_perbin_accuracy"]

        # Special rule for MLP: use mlp_mean_perbin_results.csv
        if model_id == "mlp" and mlp_mean_bin_accuracy is not None:
            row["accuracy_var"] = mlp_mean_bin_accuracy
            row["accuracy_full"] = mlp_mean_bin_accuracy

        rows.append({
            "model": model_id,
            "phase": phase,
            "attack": attack,
            "pct": pct,
            **{col: row.get(col, None) for col in METRIC_COLS}
        })

df_out = pd.DataFrame(rows)
out_path = os.path.join(BASE, f"seg{SEGMENT_ID}_all_metrics_only.csv")
df_out.to_csv(out_path, index=False)

print("Saved:", out_path)
