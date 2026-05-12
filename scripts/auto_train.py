import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from params import *
import train
from parse_config import ConfigParser

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

MODELS_TO_TRAIN = [
    "NN_1", "CNN_1", "CNN_light", "CNN_residual",
    "NN_2", "NN_3", "CNN_2", "CNN_3", "CNN_SE_Res", "tab_ft_transformer", "CNN_multiscale",
    "CNN_LSTM",
]

# MODELS_TO_TRAIN = ['tab_ft_transformer', 'CNN_multiscale']
# MODELS_TO_TRAIN = ['CNN_LSTM']
# MODELS_TO_TRAIN = ['NN_1']


# =====================================================
# =========           Functions               =========
# =====================================================

def train_model(conf_path: str):
    """Load config JSON and run training."""
    with open(conf_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    curr_config = ConfigParser(config=config_dict, resume=None, modification=None)
    train.main(curr_config)


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    for model_name in MODELS_TO_TRAIN:
        curr_conf = dict_conf[model_name]
        curr_conf_path = os.path.join(CONFIGS_DIR, f"{curr_conf}.json")
        print(f"\n=== Training {model_name} ===")
        train_model(curr_conf_path)


if __name__ == "__main__":
    main()
