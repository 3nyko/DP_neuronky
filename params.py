import json
from pathlib import Path

dict_conf = {
    "NN_1" : "config_NN_1", # první neuronka
    "CNN_1" : "config_CNN_1", # první konvolučka
    "CNN_light" : "config_CNN_light", # lehčí konvolučka
    "CNN_residual" : "config_CNN_residual",
    "NN_2" : "config_NN_2",
    "NN_3" : "config_NN_3",
    "CNN_2" : "config_CNN_2",
    "CNN_3" : "config_CNN_3", 
    "CNN_SE_Res" : "config_CNN_SE_Res", 
}

CURRENT_CONF = dict_conf['CNN_1']

_ROOT = Path(__file__).resolve().parent

# Cesta ke konfigu
DEFAULT_CONFIG_PATH = str(_ROOT / "configs" / f"{CURRENT_CONF}.json")


def find_newest_model(config_path: str) -> Path:
    """
    Latest model_best.pth under saved/models/<experiment name>/.
    Also checks legacy folder saved/models/<config file stem>/ after renames
    (e.g. name CNN_3 vs old folder config_CNN_3).
    """
    config_path = Path(config_path)
    project_root = config_path.resolve().parent.parent
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    name = cfg["name"]
    stems_to_try = []
    for stem in (name, config_path.stem):
        if stem not in stems_to_try:
            stems_to_try.append(stem)

    # Older NN_1 runs saved under this experiment name before config/name were NN_1.
    if name == "NN_1" or config_path.stem == "config_NN_1":
        if "CICIoV2024_split" not in stems_to_try:
            stems_to_try.append("CICIoV2024_split")

    for stem in stems_to_try:
        base = project_root / "saved" / "models" / stem
        if not base.is_dir():
            continue
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if not subdirs:
            continue
        latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
        candidate = latest_dir / "model_best.pth"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No model_best.pth under saved/models/{name} or saved/models/{config_path.stem}"
    )

DEFAULT_MODEL = None
try:
    DEFAULT_MODEL = str(find_newest_model(DEFAULT_CONFIG_PATH))
except Exception:
    print("Not trained yet")
