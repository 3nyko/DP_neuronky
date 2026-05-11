import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "CICIoV2024_split")

MODES = ["binary", "decimal", "hexadecimal"]
SPLITS = ["train", "val", "test"]

LABEL_COL = "specific_class"
BINARY_LABEL_COL = "label"


# =====================================================
# =========           Functions               =========
# =====================================================

def fix_split(csv_path):
    """
    Fix ALL label conflicts in a single CSV.
    For each feature pattern with multiple labels, relabel minority rows to majority.
    Returns number of rows fixed.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    feat_cols = [c for c in df.columns if c.startswith("DATA_")]
    if not feat_cols:
        print(f"    [WARN] No DATA_ columns")
        return 0

    label_col = LABEL_COL if LABEL_COL in df.columns else BINARY_LABEL_COL
    if label_col not in df.columns:
        print(f"    [WARN] No label column")
        return 0

    # Group by feature pattern, find which have more than 1 label
    grouped = df.groupby(feat_cols)[label_col].nunique()
    conflicts = grouped[grouped > 1]

    if len(conflicts) == 0:
        return 0

    total_fixed = 0

    for pattern in conflicts.index:
        if not isinstance(pattern, tuple):
            pattern = (pattern,)

        # Build mask for this pattern
        mask = pd.Series(True, index=df.index)
        for col, val in zip(feat_cols, pattern):
            mask = mask & (df[col].astype(str).str.strip() == str(val).strip())

        # Find majority label for this pattern
        label_counts = df.loc[mask, label_col].value_counts()
        majority_label = label_counts.index[0]

        # Fix rows that don't have the majority label
        wrong_mask = mask & (df[label_col] != majority_label)
        n_wrong = wrong_mask.sum()

        if n_wrong > 0:
            old_labels = df.loc[wrong_mask, label_col].value_counts().to_dict()
            print(f"    Pattern {pattern}: {old_labels} -> {majority_label} ({n_wrong} rows)")

            df.loc[wrong_mask, label_col] = majority_label

            # Also fix binary label column if present
            if BINARY_LABEL_COL in df.columns and label_col != BINARY_LABEL_COL:
                if majority_label.upper() == "BENIGN":
                    df.loc[wrong_mask, BINARY_LABEL_COL] = "BENIGN"
                else:
                    df.loc[wrong_mask, BINARY_LABEL_COL] = "ATTACK"

            total_fixed += n_wrong

    if total_fixed > 0:
        df.to_csv(csv_path, index=False)

    return total_fixed


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    """
    Fix ALL label conflicts in CICIoV2024_split dataset.
    For any feature pattern with multiple labels, minority rows get the majority label.
    Overwrites files in-place.
    """
    print(f"Data directory: {DATA_DIR}")
    print(f"Fixing label conflicts (minority -> majority)...\n")

    total_fixed = 0

    for mode in MODES:
        mode_dir = os.path.join(DATA_DIR, mode)
        if not os.path.isdir(mode_dir):
            print(f"  [SKIP] {mode_dir} does not exist")
            continue

        for split in SPLITS:
            csv_path = os.path.join(mode_dir, f"{split}.csv")
            if not os.path.isfile(csv_path):
                print(f"  [SKIP] {csv_path} not found")
                continue

            print(f"  {mode}/{split}.csv:")
            n_fixed = fix_split(csv_path)
            total_fixed += n_fixed
            if n_fixed == 0:
                print(f"    no conflicts")

    print(f"\nDone. Total rows relabeled: {total_fixed}")


if __name__ == "__main__":
    main()
