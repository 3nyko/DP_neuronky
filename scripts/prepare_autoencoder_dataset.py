import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "CICIoV2024_split")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "CICIoV2024_autoencoder")

MODES = ["binary", "decimal", "hexadecimal"]
SPLITS = ["train", "val", "test"]

LABEL_COL = "specific_class"
BENIGN_LABEL = "BENIGN"


# =====================================================
# =========           Functions               =========
# =====================================================

def prepare_mode(mode):
    """
    For a given mode (binary/decimal/hexadecimal):
    - train.csv: only BENIGN rows (autoencoder learns normal behavior)
    - val.csv: only BENIGN rows (validate reconstruction on normal data)
    - test.csv: ALL rows (benign + attacks) with labels preserved for evaluation
    """
    input_mode_dir = os.path.join(INPUT_DIR, mode)
    output_mode_dir = os.path.join(OUTPUT_DIR, mode)

    if not os.path.isdir(input_mode_dir):
        print(f"  [SKIP] {input_mode_dir} does not exist")
        return

    os.makedirs(output_mode_dir, exist_ok=True)

    for split in SPLITS:
        input_path = os.path.join(input_mode_dir, f"{split}.csv")
        output_path = os.path.join(output_mode_dir, f"{split}.csv")

        if not os.path.isfile(input_path):
            print(f"  [SKIP] {input_path} not found")
            continue

        df = pd.read_csv(input_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        original_count = len(df)

        if split in ["train", "val"]:
            # Keep only BENIGN for training/validation
            df_out = df[df[LABEL_COL].str.upper() == BENIGN_LABEL].copy()
            df_out.to_csv(output_path, index=False)
            print(f"  {mode}/{split}.csv: {len(df_out)} BENIGN rows (from {original_count} total)")
        else:
            # Test keeps everything (for anomaly evaluation)
            df.to_csv(output_path, index=False)
            benign_count = (df[LABEL_COL].str.upper() == BENIGN_LABEL).sum()
            attack_count = original_count - benign_count
            print(f"  {mode}/{split}.csv: {original_count} rows ({benign_count} benign, {attack_count} attack)")


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    """
    Prepare autoencoder dataset from existing CICIoV2024_split.

    - Train/Val: BENIGN only (autoencoder learns to reconstruct normal traffic)
    - Test: all classes (evaluate anomaly detection by reconstruction error)
    """
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    for mode in MODES:
        print(f"Mode: {mode}")
        prepare_mode(mode)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
