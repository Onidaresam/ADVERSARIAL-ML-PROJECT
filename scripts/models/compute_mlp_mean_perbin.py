import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path("/content/drive/MyDrive/ADVERSARIAL_ML_PROJECT/data/segments_time/segment_01/models_reducedscope")

# Folder where MLP per-bin metrics are stored
MLP_PERBIN_DIR = BASE_DIR / "mlp_cb_focal_research_details"

# Columns inside per-bin CSVs
ACC_COL = "accuracy_full"
F1_COL = "f1_full"

# -----------------------------
# FUNCTION
# -----------------------------

def compute_mlp_mean_perbin(regime, attack, pct):
    """
    regime ∈ {"clean_baseline", "robust", "advtrain"}
    attack ∈ {"label_flip", "sensory_add1"}
    pct    ∈ {"0_5", "1", "5", "10", "20"}
    """
    filename = f"seg01_{regime}_{attack}_{pct}_perbin_metrics.csv"
    path = MLP_PERBIN_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"MLP per-bin file not found: {path}")

    df = pd.read_csv(path)

    mean_acc = df[ACC_COL].mean()
    mean_f1 = df[F1_COL].mean()

    print(f"MLP mean per-bin metrics for {regime} | {attack} | {pct}")
    print(f"Mean Accuracy: {mean_acc:.6f}")
    print(f"Mean F1:       {mean_f1:.6f}")

    return mean_acc, mean_f1


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
# compute_mlp_mean_perbin("clean_baseline", "label_flip", "0_5")
