import json
from pathlib import Path

dict_conf = {
    "NN_1" : "config_CICIoV_split", # první neuronka
    "CNN_1" : "config_CNN_1", # první konvolučka
    "CNN_light" : "config_CNN_light", # lehčí konvolučka
    "CNN_residual" : "config_CNN_residual",
    "NN_2" : "config_NN_2",
    "NN_3" : "config_NN_3",
    "CNN_2" : "config_CNN_2",
    "CNN_3" : "config_CNN_3", 
    "CNN_SE_Res" : "config_CNN_SE_Res", 
}

CURRENT_CONF = dict_conf['CNN_SE_Res']



# Cesta ke konfigu
DEFAULT_CONFIG_PATH = f"C:\\Users\\fisar\Desktop\\Diplomka\\pytorch-template-master\\configs\\{CURRENT_CONF}.json"


# Cesta k modelu
# DEFAULT_MODEL = f"C:\\Users\\fisar\\Desktop\\Diplomka\\pytorch-template-master\\saved\\models\\{name}\\0302_123315\\model_best.pth"

def find_newest_model(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    name = config["name"]
    MODEL_BASE = Path(f"C:/Users/fisar/Desktop/Diplomka/pytorch-template-master/saved/models/{name}")
    subdirs = [p for p in MODEL_BASE.iterdir() if p.is_dir()]
    # nejnovější složka podle času úpravy
    latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
    model = latest_dir / "model_best.pth"
    return model

DEFAULT_MODEL = None
try:
    DEFAULT_MODEL = find_newest_model(DEFAULT_CONFIG_PATH)
except:
    print("Not trained yet")
