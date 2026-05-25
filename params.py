import json
import copy
from pathlib import Path

# =====================================================
# =========       Constants and options       =========
# =====================================================

_ROOT = Path(__file__).resolve().parent

CONFIGS_DICT = {
    "NN_1": "config_NN_1",
    "CNN_1": "config_CNN_1",
    "CNN_light": "config_CNN_light",
    "CNN_residual": "config_CNN_residual",
    "NN_2": "config_NN_2",
    "NN_3": "config_NN_3",
    "CNN_2": "config_CNN_2",
    "CNN_3": "config_CNN_3",
    "CNN_SE_Res": "config_CNN_SE_Res",
    "tab_ft_transformer": "config_tab_ft_transformer",
    "CNN_multiscale": "config_CNN_multiscale",
    "CNN_LSTM": "config_CNN_LSTM",
}

dict_conf = CONFIGS_DICT

# Dataset switch: "ciciov" | "car"
# Car-Hacking: set CURRENT_DATASET = "car", run prepare_car_hacking_dataset.py first.
DATASETS = {
    "ciciov": {
        "data_dir": "data/CICIoV2024_split",
        "data_loader": "CICIoV2024_DataLoader",
        "mode": "hexadecimal",
        "name_suffix": "",
    },
    "car": {
        "data_dir": "data/Car_hacking_split",
        "data_loader": "Car_Hacking_DataLoader",
        "mode": None,
        "name_suffix": "_car",
    },
}

CURRENT_DATASET = "ciciov"
CURRENT_MODEL = "CNN_light"

AUTOENCODER_CONFIGS_DICT = {
    "shallow": "config_autoencoder_shallow",
    "deep": "config_autoencoder_deep",
}

CURRENT_AUTOENCODER = "shallow"
DEFAULT_AUTOENCODER_CONFIG_PATH = str(
    _ROOT / "configs" / f"{AUTOENCODER_CONFIGS_DICT[CURRENT_AUTOENCODER]}.json"
)


# =====================================================
# =========           Functions               =========
# =====================================================

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def build_config(model: str | None = None, dataset: str | None = None) -> dict:
    """Build config dict from base model JSON + dataset settings."""
    model = model or CURRENT_MODEL
    dataset = dataset or CURRENT_DATASET

    if model not in CONFIGS_DICT:
        valid = ", ".join(sorted(CONFIGS_DICT))
        raise KeyError(f"Unknown model '{model}'. Choose from: {valid}")
    if dataset not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from: {valid}")

    cfg = copy.deepcopy(_read_json(_ROOT / "configs" / f"{CONFIGS_DICT[model]}.json"))
    ds = DATASETS[dataset]

    cfg["name"] = f"{model}{ds['name_suffix']}"
    cfg["data_loader"]["type"] = ds["data_loader"]
    cfg["data_loader"]["args"]["data_dir"] = ds["data_dir"]

    if ds["mode"] is not None:
        cfg["data_loader"]["args"]["mode"] = ds["mode"]
    else:
        cfg["data_loader"]["args"].pop("mode", None)

    # Car CSVs are large; multiprocessing workers fail on Windows (spawn/pickle).
    if dataset == "car":
        cfg["data_loader"]["args"]["num_workers"] = 0

    return cfg


def model_config_path(model: str | None = None, dataset: str | None = None) -> Path:
    """JSON path for train.py / test.py -c (CICIoV static, Car generated)."""
    model = model or CURRENT_MODEL
    dataset = dataset or CURRENT_DATASET
    stem = CONFIGS_DICT[model]
    if dataset == "car":
        return _ROOT / "configs" / f"{stem}_car.json"
    return _ROOT / "configs" / f"{stem}.json"


def ensure_model_config(model: str | None = None, dataset: str | None = None) -> Path:
    """
    Return config JSON path. For Car-Hacking writes/updates configs/<model>_car.json.
  """
    model = model or CURRENT_MODEL
    dataset = dataset or CURRENT_DATASET
    path = model_config_path(model, dataset)

    if dataset == "car":
        _write_json(path, build_config(model, dataset))
    elif not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    return path


def autoencoder_config_path(model=None, config_path=None) -> Path:
    if config_path is not None:
        return Path(config_path)
    key = model or CURRENT_AUTOENCODER
    if key not in AUTOENCODER_CONFIGS_DICT:
        valid = ", ".join(sorted(AUTOENCODER_CONFIGS_DICT))
        raise KeyError(f"Unknown autoencoder '{key}'. Choose from: {valid}")
    return _ROOT / "configs" / f"{AUTOENCODER_CONFIGS_DICT[key]}.json"


def find_newest_model(model: str | None = None, dataset: str | None = None) -> Path:
    model = model or CURRENT_MODEL
    dataset = dataset or CURRENT_DATASET
    name = build_config(model, dataset)["name"]

    stems_to_try = []
    for stem in (name, CONFIGS_DICT[model], f"{CONFIGS_DICT[model]}_car"):
        if stem not in stems_to_try:
            stems_to_try.append(stem)

    if dataset == "ciciov" and model == "NN_1":
        if "CICIoV2024_split" not in stems_to_try:
            stems_to_try.append("CICIoV2024_split")

    for stem in stems_to_try:
        base = _ROOT / "saved" / "models" / stem
        if not base.is_dir():
            continue
        subdirs = sorted(
            [p for p in base.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for subdir in subdirs:
            candidate = subdir / "model_best.pth"
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"No model_best.pth for model={model} dataset={dataset} (experiment name: {name})"
    )


# =====================================================
# =========          Defaults (train/test)    =========
# =====================================================

CURRENT_CONF = CONFIGS_DICT[CURRENT_MODEL]
DEFAULT_CONFIG_PATH = str(ensure_model_config(CURRENT_MODEL, CURRENT_DATASET))

DEFAULT_MODEL = None
try:
    DEFAULT_MODEL = str(find_newest_model(CURRENT_MODEL, CURRENT_DATASET))
except Exception:
    print("Not trained yet")
