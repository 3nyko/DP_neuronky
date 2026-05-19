import os
import sys
import json
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from params import *
from parse_config import ConfigParser

test_module = importlib.import_module("test")

# =====================================================
# =========       Constants and options       =========
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

MODELS_TO_TEST = [
    "NN_1", "CNN_1", "CNN_light", "CNN_residual",
    "NN_2", "NN_3", "CNN_2", "CNN_3", "CNN_SE_Res", "tab_ft_transformer", "CNN_multiscale",
    "CNN_LSTM",
]


# =====================================================
# =========           Functions               =========
# =====================================================

def test_model(conf_path: str):
    """Load config JSON, find best checkpoint, and run testing."""
    resume_path = str(find_newest_model(conf_path))
    print(f"  Checkpoint: {resume_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    curr_config = ConfigParser(config=config_dict, resume=resume_path, modification=None)
    test_module.main(curr_config)


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    for model_name in MODELS_TO_TEST:
        curr_conf = dict_conf[model_name]
        curr_conf_path = os.path.join(CONFIGS_DIR, f"{curr_conf}.json")
        print(f"\n=== Testing {model_name} ===")
        try:
            test_model(curr_conf_path)
        except FileNotFoundError as e:
            print(f"  SKIPPED (no checkpoint): {e}")
        except RuntimeError as e:
            print(f"  SKIPPED (incompatible checkpoint): {e}")


if __name__ == "__main__":
    main()
