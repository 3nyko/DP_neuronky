import os
import sys
import json
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from params import CURRENT_DATASET, ensure_model_config, find_newest_model
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

def test_model(model_name: str):
    """Load config JSON, find best checkpoint, and run testing."""
    conf_path = ensure_model_config(model_name, CURRENT_DATASET)
    resume_path = str(find_newest_model(model_name, CURRENT_DATASET))
    print(f"  Config: {conf_path}")
    print(f"  Checkpoint: {resume_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    curr_config = ConfigParser(config=config_dict, resume=resume_path, modification=None)
    test_module.main(curr_config)


# =====================================================
# =========              Main                 =========
# =====================================================

def main():
    print(f"Dataset: {CURRENT_DATASET}")
    for model_name in MODELS_TO_TEST:
        print(f"\n=== Testing {model_name} ({CURRENT_DATASET}) ===")
        try:
            test_model(model_name)
        except FileNotFoundError as e:
            print(f"  SKIPPED (no checkpoint): {e}")
        except RuntimeError as e:
            print(f"  SKIPPED (incompatible checkpoint): {e}")


if __name__ == "__main__":
    main()
