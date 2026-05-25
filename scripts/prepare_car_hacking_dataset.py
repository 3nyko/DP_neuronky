"""
Prepare Car-Hacking Dataset for use with the project data loaders.

Pipeline:
  1) Parse raw logs (hex CAN ID + payload) -> integer ID + DATA_0..DATA_7
  2) Per-class CSVs in data/Car_hacking/
  3) Split 70/20/10 + shuffle -> data/Car_hacking_split/train|val|test.csv
  4) Optional: fix label conflicts, build autoencoder set (BENIGN-only train/val)

Raw format (no header): timestamp, CAN_ID, DLC, D0..D7, R

Note: CICIoV2024 uses binary/decimal/hexadecimal variants because the dataset
is published that way. Car-Hacking is converted once to decimal integers only.
In config JSON set data_loader.args.mode to \"decimal\".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "Car-Hacking Dataset"
DEFAULT_CONVERTED_DIR = PROJECT_ROOT / "data" / "Car_hacking"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "Car_hacking_split"
DEFAULT_AE_DIR = PROJECT_ROOT / "data" / "Car_hacking_autoencoder"

SPLITS = ["train", "val", "test"]
OUTPUT_COLUMNS = (
    ["ID"] + [f"DATA_{i}" for i in range(8)] + ["label", "category", "specific_class"]
)

VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
TRAIN_SPLIT = 1.0 - (VAL_SPLIT + TEST_SPLIT)
RANDOM_STATE = 200701
CHUNK_SIZE = 100_000

RAW_COLUMNS = [
    "timestamp",
    "can_id",
    "dlc",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "flag",
]

CAR_FILE_MAP = {
    "normal_run_data.csv": ("benign", "BENIGN", "BENIGN"),
    "DoS_dataset.csv": ("DoS", "ATTACK", "DoS"),
    "Fuzzy_dataset.csv": ("Fuzzy", "ATTACK", "Fuzzy"),
    "gear_dataset.csv": ("Gear", "ATTACK", "Gear"),
    "RPM_dataset.csv": ("RPM", "ATTACK", "RPM"),
}

LABEL_COL = "specific_class"
BINARY_LABEL_COL = "label"

# =====================================================
# =========           Parsing                 =========
# =====================================================


def parse_hex_int(raw_val) -> int:
    s = str(raw_val).strip().upper()
    if s.startswith("0X"):
        s = s[2:]
    return int(s, 16)


def row_to_record(can_id: int, data_bytes: list[int], label: str, specific: str) -> dict:
    return {
        "ID": can_id,
        **{f"DATA_{i}": data_bytes[i] for i in range(8)},
        BINARY_LABEL_COL: label,
        "category": specific,
        LABEL_COL: specific,
    }


def convert_chunk(chunk: pd.DataFrame, label: str, specific: str) -> pd.DataFrame:
    rows = []
    for _, r in chunk.iterrows():
        try:
            can_id = parse_hex_int(r["can_id"])
            data_bytes = [parse_hex_int(r[f"d{i}"]) for i in range(8)]
        except (ValueError, TypeError):
            continue
        rows.append(row_to_record(can_id, data_bytes, label, specific))
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def read_raw_chunks(csv_path: Path):
    yield from pd.read_csv(
        csv_path,
        header=None,
        names=RAW_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =====================================================
# =========        Step 1: conversion         =========
# =====================================================


def convert_raw_file(raw_path: Path, stem: str, label: str, specific: str, out_dir: Path) -> int:
    out_path = out_dir / f"{stem}.csv"
    if out_path.exists():
        out_path.unlink()

    total = 0
    write_header = True
    for chunk in read_raw_chunks(raw_path):
        converted = convert_chunk(chunk, label, specific)
        if converted.empty:
            continue
        converted.to_csv(out_path, mode="a", index=False, header=write_header)
        write_header = False
        total += len(converted)

    print(f"    {out_path.name}: {total:,} rows")
    return total


def filter_file_map(only_stems: list[str] | None) -> dict:
    if not only_stems:
        return CAR_FILE_MAP
    wanted = {s.lower() for s in only_stems}
    selected = {k: v for k, v in CAR_FILE_MAP.items() if v[0].lower() in wanted}
    if not selected:
        valid = ", ".join(v[0] for v in CAR_FILE_MAP.values())
        raise ValueError(f"--only: unknown stem(s). Choose from: {valid}")
    return selected


def convert_all_raw(raw_dir: Path, converted_dir: Path, only_stems: list[str] | None = None) -> None:
    file_map = filter_file_map(only_stems)
    ensure_dir(converted_dir)
    print(f"Converting raw data\n  from: {raw_dir}\n  to:   {converted_dir}\n")

    for raw_name, (stem, label, specific) in file_map.items():
        raw_path = raw_dir / raw_name
        if not raw_path.is_file():
            print(f"  [SKIP] missing {raw_path}")
            continue
        print(f"  {raw_name} ({specific})")
        convert_raw_file(raw_path, stem, label, specific, converted_dir)

    print()


# =====================================================
# =========        Step 2: train/val/test     =========
# =====================================================


def split_dataset(converted_dir: Path, split_dir: Path) -> None:
    all_files = sorted(converted_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files in {converted_dir}")

    print(
        f"Splitting (train {TRAIN_SPLIT:.0%} / val {VAL_SPLIT:.0%} / test {TEST_SPLIT:.0%}, "
        f"seed={RANDOM_STATE})\n  from: {converted_dir}\n  to:   {split_dir}\n"
    )

    train_parts, val_parts, test_parts = [], [], []

    for fpath in all_files:
        df = pd.read_csv(fpath, low_memory=False)
        if df.empty:
            print(f"  [SKIP] empty {fpath.name}")
            continue

        trainval_df, test_df = train_test_split(
            df, test_size=TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True
        )
        relative_val_split = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=relative_val_split,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)
        print(f"  {fpath.name}: {len(df):,} rows")

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    ensure_dir(split_dir)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(
        f"\n  -> train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}\n"
    )


# =====================================================
# =========     Step 3: label conflict fix    =========
# =====================================================


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    if "ID" in df.columns:
        cols.append("ID")
    cols.extend([c for c in df.columns if re.match(r"^DATA_\d+$", c)])
    return cols


def fix_split_csv(csv_path: Path) -> int:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    feat_cols = feature_columns(df)
    label_col = LABEL_COL if LABEL_COL in df.columns else BINARY_LABEL_COL
    if not feat_cols or label_col not in df.columns:
        return 0

    grouped = df.groupby(feat_cols)[label_col].nunique()
    conflicts = grouped[grouped > 1]
    if len(conflicts) == 0:
        return 0

    total_fixed = 0
    for pattern in conflicts.index:
        if not isinstance(pattern, tuple):
            pattern = (pattern,)

        mask = pd.Series(True, index=df.index)
        for col, val in zip(feat_cols, pattern):
            mask &= df[col].astype(str).str.strip() == str(val).strip()

        majority_label = df.loc[mask, label_col].value_counts().index[0]
        wrong_mask = mask & (df[label_col] != majority_label)
        n_wrong = int(wrong_mask.sum())
        if n_wrong <= 0:
            continue

        df.loc[wrong_mask, label_col] = majority_label
        if BINARY_LABEL_COL in df.columns and label_col != BINARY_LABEL_COL:
            df.loc[wrong_mask, BINARY_LABEL_COL] = (
                "BENIGN" if str(majority_label).upper() == "BENIGN" else "ATTACK"
            )
        total_fixed += n_wrong

    if total_fixed > 0:
        df.to_csv(csv_path, index=False)
    return total_fixed


def fix_all_splits(split_dir: Path) -> None:
    print(f"Fixing label conflicts in {split_dir}\n")
    total = 0
    for split in SPLITS:
        csv_path = split_dir / f"{split}.csv"
        if not csv_path.is_file():
            continue
        n = fix_split_csv(csv_path)
        total += n
        print(f"  {split}.csv: {n:,} rows relabeled")
    print(f"  Total relabeled: {total:,}\n")


# =====================================================
# =========   Step 4: autoencoder dataset     =========
# =====================================================


def prepare_autoencoder(split_dir: Path, ae_dir: Path) -> None:
    print(f"Autoencoder dataset\n  from: {split_dir}\n  to:   {ae_dir}\n")
    ensure_dir(ae_dir)

    for split in SPLITS:
        src = split_dir / f"{split}.csv"
        dst = ae_dir / f"{split}.csv"
        if not src.is_file():
            continue

        df = pd.read_csv(src, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        n_all = len(df)

        if split in ("train", "val"):
            df = df[df[LABEL_COL].astype(str).str.upper() == "BENIGN"].copy()

        df.to_csv(dst, index=False)

        if split in ("train", "val"):
            print(f"  {split}.csv: {len(df):,} BENIGN (from {n_all:,})")
        else:
            benign = (df[LABEL_COL].astype(str).str.upper() == "BENIGN").sum()
            print(f"  {split}.csv: {n_all:,} rows ({benign:,} benign, {n_all - benign:,} attack)")
    print()


# =====================================================
# =========              Main                 =========
# =====================================================


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Car-Hacking dataset (parse, split, shuffle).")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--converted-dir", type=Path, default=DEFAULT_CONVERTED_DIR)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--ae-dir", type=Path, default=DEFAULT_AE_DIR)
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--fix-conflicts", action="store_true")
    p.add_argument("--autoencoder", action="store_true")
    p.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        help="benign, DoS, Fuzzy, Gear, RPM",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_convert:
        if not args.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")
        convert_all_raw(args.raw_dir, args.converted_dir, args.only)

    if not args.skip_split:
        if not args.converted_dir.is_dir():
            raise FileNotFoundError(f"Converted directory not found: {args.converted_dir}")
        split_dataset(args.converted_dir, args.split_dir)

    if args.fix_conflicts:
        fix_all_splits(args.split_dir)

    if args.autoencoder:
        prepare_autoencoder(args.split_dir, args.ae_dir)

    print("Done.")
    print(f"  per-class: {args.converted_dir}")
    print(f"  split:     {args.split_dir}")
    if args.autoencoder:
        print(f"  autoencoder: {args.ae_dir}")
    print('  Training config: data_dir + mode "decimal"')


if __name__ == "__main__":
    main()
