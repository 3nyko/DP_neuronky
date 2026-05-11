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


# =====================================================
# =========           Functions               =========
# =====================================================

def print_class_distribution(df):
    """Print row counts per class."""
    print(f"\n  Class distribution:")
    for cls, cnt in df[LABEL_COL].value_counts().items():
        print(f"    {cls:20s}: {cnt:>8,} rows")


def print_unique_patterns(df, feat_cols):
    """Print unique feature vector count per class."""
    print(f"\n  Unique feature patterns per class:")
    for cls in sorted(df[LABEL_COL].unique()):
        subset = df[df[LABEL_COL] == cls]
        uniq = subset[feat_cols].drop_duplicates()
        print(f"    {cls:20s}: {len(uniq):>6,} unique patterns  ({len(subset):,} rows)")


def print_label_conflicts(df, feat_cols, n):
    """Detect and print label conflicts (same features, different labels)."""
    grouped = df.groupby(feat_cols)[LABEL_COL].nunique()
    conflicts = grouped[grouped > 1]

    print(f"\n  Label conflicts (same features, different label): {len(conflicts)} pattern(s)")

    if len(conflicts) == 0:
        return

    conflict_keys = conflicts.index
    mask = df.set_index(feat_cols).index.isin(conflict_keys)
    conflict_rows = df[mask]

    print(f"  Total rows involved in conflicts: {mask.sum():,}")
    print(f"\n  Conflict breakdown by class:")
    for cls, cnt in conflict_rows[LABEL_COL].value_counts().items():
        print(f"    {cls:20s}: {cnt:>8,} rows")

    print(f"\n  Conflicting patterns detail:")
    for i, key in enumerate(list(conflict_keys)[:20]):
        if not isinstance(key, tuple):
            key = (key,)
        subset = df
        for col, v in zip(feat_cols, key):
            subset = subset[subset[col] == v]
        pattern_str = " ".join(str(v) for v in key)
        labels = subset[LABEL_COL].value_counts().to_dict()
        print(f"    [{pattern_str}] => {labels}")

    # Accuracy ceiling: for each conflict, the best model predicts majority
    min_errors = 0
    for key in conflict_keys:
        if not isinstance(key, tuple):
            key = (key,)
        subset = df
        for col, v in zip(feat_cols, key):
            subset = subset[subset[col] == v]
        counts = subset[LABEL_COL].value_counts()
        min_errors += counts.sum() - counts.max()

    ceiling = (n - min_errors) / n * 100
    print(f"\n  >> Theoretical accuracy ceiling (majority-vote): {ceiling:.10f}%")
    print(f"  >> Minimum unavoidable errors: {min_errors}")


def print_cross_class_overlap(df, feat_cols):
    """Print feature patterns shared between different classes."""
    print(f"\n  Cross-class pattern overlap:")
    classes = sorted(df[LABEL_COL].unique())
    any_overlap = False
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1:]:
            s1 = df[df[LABEL_COL] == c1][feat_cols].drop_duplicates()
            s2 = df[df[LABEL_COL] == c2][feat_cols].drop_duplicates()
            merged = s1.merge(s2, on=feat_cols)
            if len(merged) > 0:
                any_overlap = True
                print(f"    {c1} vs {c2}: {len(merged)} shared pattern(s)")
    if not any_overlap:
        print(f"    (none)")


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    """
    Analyze the dataset for label conflicts and pattern diversity.

    Checks for:
    - Duplicate feature vectors with conflicting labels (same DATA_0..7, different class)
    - Number of unique feature patterns per class
    - Cross-class pattern overlap
    - Accuracy ceiling estimation based on conflicts
    """
    for mode in MODES:
        mode_dir = os.path.join(DATA_DIR, mode)
        if not os.path.isdir(mode_dir):
            print(f"[SKIP] {mode_dir} does not exist")
            continue

        for split in SPLITS:
            csv_path = os.path.join(mode_dir, f"{split}.csv")
            if not os.path.isfile(csv_path):
                print(f"[SKIP] {csv_path} not found")
                continue

            print(f"\n{'='*60}")
            print(f"  Mode: {mode.upper()} | Split: {split.upper()}")
            print(f"  ({csv_path})")
            print(f"{'='*60}")

            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            feat_cols = [c for c in df.columns if c.startswith("DATA_")]
            n = len(df)
            print(f"\n  Total rows: {n:,}")
            print(f"  Feature columns: {feat_cols}")

            print_class_distribution(df)
            print_unique_patterns(df, feat_cols)
            print_label_conflicts(df, feat_cols, n)
            print_cross_class_overlap(df, feat_cols)


if __name__ == "__main__":
    main()
