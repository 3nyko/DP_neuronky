import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from enum import Enum

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =====================================================
# =========       Constants and options       =========
# =====================================================

class Mode(Enum):
    BINARY = "binary"
    DECIMAL = "decimal"
    HEXADECIMAL = "hexadecimal"

DEFAULT_MODE = Mode.HEXADECIMAL 

# for multiclass classification
USE_MULTICLASS = True

BINCLASS_DICT = {
    "BENIGN" : 0,
    "ATTACK" : 1,
}
MULTICLASS_DICT = {
    "BENIGN" : 0,
    "DOS" : 1,
    "GAS" : 2,
    "RPM" : 3,
    "SPEED" : 4,
    "STEERING_WHEEL" : 5
}

# Car-Hacking Dataset (prepare_car_hacking_dataset.py)
CAR_MULTICLASS_DICT = {
    "BENIGN": 0,
    "DOS": 1,
    "FUZZY": 2,
    "GEAR": 3,
    "RPM": 4,
}


# ======================================================
# =========             Dataset              ===========
# ======================================================

class CICIoV2024_Dataset(Dataset):
    """
    Dataset loader for already split CICIoV2024 dataset
    """

    def __init__(self, data_dir, mode=DEFAULT_MODE, split="train", multiclass = False):
        self.mode = mode

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")

        self.split = split

        # cesta k csv souboru
        self.file_path = os.path.join(data_dir, self.mode.value, f"{split}.csv")
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # načti data
        self.data = pd.read_csv(self.file_path, low_memory=False)
        self.data.columns = [c.strip() for c in self.data.columns]  # normalize column names

        # vyber pouze datové sloupce
        self.data_no_labels = self._remove_labels(self.data)
        if multiclass: label_data = self.data["specific_class"]
        else: label_data = self.data["label"]
        
        self.labels = self._map_labels(label_data, multiclass)

    def _remove_labels(self, df):
        """Return data columns (ID + DATA_0 ... DATA_7)"""
        feature_cols = []
        if "ID" in df.columns:
            feature_cols.append("ID")
        data_cols = [c for c in df.columns if c.startswith("DATA_")]
        if not data_cols:
            raise ValueError("No DATA_ columns found in dataset.")
        feature_cols.extend(data_cols)
        return df[feature_cols].values

    def _map_labels(self, labels, multiclass = False):
        """BENIGN = 0, ATTACK = 1"""
        if multiclass: class_dict = MULTICLASS_DICT
        else: class_dict = BINCLASS_DICT
        return(labels.str.upper().map(class_dict))

    def __len__(self):
        return len(self.data_no_labels)

    def __getitem__(self, idx):
        data = self.data_no_labels[idx].copy()

        def safe_int(val, base):
            # Pandas často čte hodnoty jako float → přetypujeme na string
            if isinstance(val, (int, float)): # if number, just return
                return int(val)
            s = str(val).strip()
            if "." in s and s.replace(".", "").isdigit(): # for broken HEX values like 2.0 -> 2
                return int(float(s))
            return int(s, base)

        # Převod podle módu
        if self.mode in [Mode.BINARY, Mode.BINARY.value]:
            data = [safe_int(v, base=2) for v in data]
        elif self.mode in [Mode.HEXADECIMAL, Mode.HEXADECIMAL.value]:
            data = [safe_int(v, base=16) for v in data]
        elif self.mode in [Mode.DECIMAL, Mode.DECIMAL.value]:
            data = [float(v) if not pd.isna(v) else 0.0 for v in data]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, labels


class CICIoV2024_Dataset_no_split(Dataset):
    """
    Loads all CSVs from the given CICIoV2024 mode folder (binary/decimal/hexadecimal).
    Each CSV must contain DATA_0 ... DATA_7 columns and 'label'.
    """
    def __init__(self, data_dir, mode=DEFAULT_MODE):
        self.mode = mode
        self.folder = os.path.join(data_dir, mode)
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        self.data = self._load_all_csvs(self.folder)
        self.data_no_labels = self._remove_labels(self.data)
        self.labels = self._map_labels(self.data["label"])

    def _load_all_csvs(self, folder_path):
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")

        df_list = []
        for f in all_files:
            df = pd.read_csv(f, low_memory=False)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df.columns = [c.strip() for c in df.columns] # normalize column names
        # ensure numeric conversion for data columns
        for col in df.columns:
            if col.startswith("DATA_") or col in ["ID", "DLC"]:
                df[col] = pd.to_numeric(df[col], errors="coerce") # TODO: Overit funkcnost pro Binary/Hexadec
        df = df.fillna(0) # vyplnit pripadne prazdne hodnoty nulou
        return df

    def _remove_labels(self, df):
        """Return only numeric feature columns."""
        feature_cols = [c for c in df.columns if c.startswith("DATA_")]
        if not feature_cols:
            raise ValueError("No DATA_ columns found in dataset.")
        return df[feature_cols].values

    def _map_labels(self, labels):
        """(0=BENIGN, 1=ATTACK)."""
        mapped = labels.str.upper().map(lambda x: 0 if "BENIGN" in x else 1)
        return mapped.values

    def __len__(self):
        return len(self.data_no_labels)

    def __getitem__(self, idx):
        x = self.data_no_labels[idx].copy()

        def safe_int(val, base=10):
            return int(str(val), base) # Pandas často čte hodnoty jako float → přetypujeme na string

        # Převod podle módu
        if self.mode in [Mode.BINARY, Mode.BINARY.value]:
            x = [safe_int(v, base=2) for v in x]
        elif self.mode in [Mode.HEXADECIMAL, Mode.HEXADECIMAL.value]:
            x = [safe_int(v, base=16) for v in x]
        elif self.mode in [Mode.DECIMAL, Mode.DECIMAL.value]:
            x = [float(v) if not pd.isna(v) else 0.0 for v in x]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ========================================================
# =========             DataLoader              ==========
# ========================================================

class CICIoV2024_DataLoader:
    """
    DataLoader for CICIoV2024_split dataset (uses predefined train.csv and val.csv instead of random split)
    """
    def __init__(self, data_dir, batch_size, mode=DEFAULT_MODE, num_workers=2, shuffle=True, split="train", multiclass=USE_MULTICLASS):
        self.batch_size = batch_size
        # dataset
        dataset = CICIoV2024_Dataset(data_dir=data_dir, mode=mode, split=split, multiclass=multiclass)
        shuff = shuffle if split=="train" else False
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuff, num_workers=num_workers
        )

        # valid loader
        self.valid_data_loader = None
        if split == "train":
            val_dataset = CICIoV2024_Dataset(data_dir=data_dir, mode=mode, split="val", multiclass=multiclass)
            self.valid_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        self.input_dim = dataset[0][0].shape[-1] # velikost dat (pro vstupni neur vrstvu) - 8
        self.mode = mode

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def split_validation(self):
        """Return validation DataLoader"""
        return self.valid_data_loader

def _safe_num_workers(num_workers: int) -> int:
    """Windows spawn cannot pickle huge pandas datasets used by workers."""
    if sys.platform == "win32" and num_workers > 0:
        return 0
    return num_workers


class Car_Hacking_Dataset(Dataset):
    """
    Split Car-Hacking CSVs (flat layout: data_dir/train.csv, val.csv, test.csv).
    Features are decimal integers: ID + DATA_0..DATA_7.
    """

    def __init__(self, data_dir, split="train", multiclass=USE_MULTICLASS):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.file_path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        df = pd.read_csv(self.file_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        if multiclass:
            label_data = df["specific_class"]
        else:
            label_data = df["label"]

        self.features = np.ascontiguousarray(
            self._extract_features(df), dtype=np.float32
        )
        self.labels = self._map_labels(label_data, multiclass)

    def _extract_features(self, df):
        feature_cols = []
        if "ID" in df.columns:
            feature_cols.append("ID")
        data_cols = sorted(
            [c for c in df.columns if c.startswith("DATA_")],
            key=lambda c: int(c.split("_")[1]),
        )
        if not data_cols:
            raise ValueError("No DATA_ columns found in dataset.")
        feature_cols.extend(data_cols)
        return df[feature_cols].to_numpy()

    def _map_labels(self, labels, multiclass=False):
        class_dict = CAR_MULTICLASS_DICT if multiclass else BINCLASS_DICT
        mapped = labels.astype(str).str.strip().str.upper().map(class_dict)
        if mapped.isna().any():
            unknown = sorted(labels[mapped.isna()].astype(str).unique())
            raise ValueError(f"Unknown class label(s): {unknown}")
        return mapped.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Car_Hacking_DataLoader:
    """DataLoader for data/Car_hacking_split (train.csv / val.csv / test.csv)."""

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers=2,
        shuffle=True,
        split="train",
        multiclass=USE_MULTICLASS,
        **kwargs,
    ):
        kwargs.pop("mode", None)
        num_workers = _safe_num_workers(num_workers)
        self.batch_size = batch_size
        dataset = Car_Hacking_Dataset(
            data_dir=data_dir, split=split, multiclass=multiclass
        )
        shuff = shuffle if split == "train" else False
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuff, num_workers=num_workers
        )

        self.valid_data_loader = None
        if split == "train":
            val_dataset = Car_Hacking_Dataset(
                data_dir=data_dir, split="val", multiclass=multiclass
            )
            self.valid_data_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        self.input_dim = dataset[0][0].shape[-1]

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def split_validation(self):
        return self.valid_data_loader


class CICIoV2024_DataLoader_no_split(BaseDataLoader):
    """
    DataLoader for CICIoV2024 datasets.
    """
    def __init__(self, data_dir, batch_size, mode=DEFAULT_MODE,
                 shuffle=True, validation_split=0.1, num_workers=2):
        dataset = CICIoV2024_Dataset_no_split(data_dir=data_dir, mode=mode)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)


# ========================================================
# =========       Autoencoder Dataset/Loader     =========
# ========================================================

class CICIoV2024_Autoencoder_Dataset(Dataset):
    """
    Dataset for autoencoder training.
    Returns (features, features) — input equals target (reconstruction task).
    Labels are stored separately for evaluation but not used during training.
    """

    def __init__(self, data_dir, mode=DEFAULT_MODE, split="train"):
        self.mode = mode
        self.split = split

        self.file_path = os.path.join(data_dir, self.mode.value, f"{split}.csv")
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.data = pd.read_csv(self.file_path, low_memory=False)
        self.data.columns = [c.strip() for c in self.data.columns]

        # feature columns (ID + DATA_0..7)
        feature_cols = []
        if "ID" in self.data.columns:
            feature_cols.append("ID")
        data_cols = [c for c in self.data.columns if c.startswith("DATA_")]
        if not data_cols:
            raise ValueError("No DATA_ columns found in dataset.")
        feature_cols.extend(data_cols)
        self.feature_values = self.data[feature_cols].values

        # labels for evaluation (binary: 0=benign, 1=attack)
        if "label" in self.data.columns:
            self.labels = self.data["label"].str.upper().map(
                lambda x: 0 if "BENIGN" in x else 1
            ).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.feature_values)

    def __getitem__(self, idx):
        data = self.feature_values[idx].copy()

        def safe_int(val, base):
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val).strip()
            if "." in s and s.replace(".", "").isdigit():
                return int(float(s))
            return int(s, base)

        if self.mode in [Mode.BINARY, Mode.BINARY.value]:
            data = [safe_int(v, base=2) for v in data]
        elif self.mode in [Mode.HEXADECIMAL, Mode.HEXADECIMAL.value]:
            data = [safe_int(v, base=16) for v in data]
        elif self.mode in [Mode.DECIMAL, Mode.DECIMAL.value]:
            data = [float(v) if not pd.isna(v) else 0.0 for v in data]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        x = torch.tensor(data, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, label
        return x, x


class CICIoV2024_Autoencoder_DataLoader:
    """
    DataLoader for autoencoder training on CICIoV2024_autoencoder dataset.
    Train/val contain only BENIGN; test contains all classes.
    """

    def __init__(self, data_dir, batch_size, mode=DEFAULT_MODE, num_workers=2, shuffle=True, split="train"):
        self.batch_size = batch_size

        dataset = CICIoV2024_Autoencoder_Dataset(data_dir=data_dir, mode=mode, split=split)
        shuff = shuffle if split == "train" else False
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuff, num_workers=num_workers
        )

        self.valid_data_loader = None
        if split == "train":
            val_dataset = CICIoV2024_Autoencoder_Dataset(data_dir=data_dir, mode=mode, split="val")
            self.valid_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        self.input_dim = dataset[0][0].shape[-1]
        self.mode = mode

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def split_validation(self):
        """Return validation DataLoader"""
        return self.valid_data_loader