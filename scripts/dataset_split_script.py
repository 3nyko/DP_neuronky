import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# =========       Constants and options       =========
# =====================================================

DATASET_PATH = r"C:\Users\fisar\Desktop\Diplomka\pytorch-template-master\data\CICIoV2024"
OUTPUT_PATH = r"C:\Users\fisar\Desktop\Diplomka\pytorch-template-master\data\CICIoV2024_split"

MODES = ["binary", "decimal", "hexadecimal"]

VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
TRAIN_SPLIT = 1 - (VAL_SPLIT + TEST_SPLIT)
RANDOM_STATE = 200701


# =====================================================
# =========           Functions               =========
# =====================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def process_mode(mode):
    """Split each CSV first, then merge train/val/test parts and save."""
    mode_dir = os.path.join(DATASET_PATH, mode)
    if not os.path.isdir(mode_dir):
        print(f"Missing folder: {mode_dir}")
        return

    all_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith(".csv")]
    if not all_files:
        print(f"No CSV files found in {mode_dir}")
        return

    print(f"Processing mode: {mode} ({len(all_files)} files)")

    # Každý CSV rozdělíme samostatně na train/val/test
    train_parts = []
    val_parts = []
    test_parts = []

    for fpath in all_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if df.empty:
                print(f"Skipping empty file: {fpath}")
                continue

            # Split 1: odděl test část
            trainval_df, test_df = train_test_split(
                df, test_size=TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True
            )

            # Split 2: z trainval odděl validation tak, aby celkové poměry zůstaly stejné
            relative_val_split = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
            train_df, val_df = train_test_split(
                trainval_df, test_size=relative_val_split, random_state=RANDOM_STATE, shuffle=True
            )

            train_parts.append(train_df)
            val_parts.append(val_df)
            test_parts.append(test_df)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not train_parts or not val_parts or not test_parts:
        print(f"No usable CSV data found in {mode_dir}")
        return

    # Sloučení stejných částí napříč všemi CSV
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Finální shuffle každé části
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Combined split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Uložení do výstupních CSV
    out_dir = os.path.join(OUTPUT_PATH, mode)
    ensure_dir(out_dir)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved: train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    """Split raw CICIoV2024 CSVs into train/val/test for each mode."""
    for mode in MODES:
        process_mode(mode)

    print("\nAll splits completed and saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
