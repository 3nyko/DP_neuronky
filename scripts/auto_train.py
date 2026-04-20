import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from params import *
import train
from parse_config import ConfigParser


# MODELS_TO_TRAIN = ['NN_2', 'NN_3', 'CNN_2', 'CNN_3', 'CNN_SE_Res']
MODELS_TO_TRAIN = ['CNN_SE_Res']


def train_model(conf_path: str):
    with open(conf_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    curr_config = ConfigParser(config=config_dict, resume=None, modification=None)
    train.main(curr_config)


def main():

    for model_name in MODELS_TO_TRAIN:
        curr_conf = dict_conf[model_name]
        curr_conf_path = f"C:\\Users\\fisar\\Desktop\\Diplomka\\pytorch-template-master\\configs\\{curr_conf}.json"
        print(f"\n=== Training {model_name} ===")
        train_model(curr_conf_path)


if __name__ == '__main__':
    main()