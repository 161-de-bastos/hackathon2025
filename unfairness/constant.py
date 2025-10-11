import json
import os

KB_CATEGORIES = ("A", "CH", "CR", "LTD", "TER")

def kb_file_for(cat):
    return os.path.join("kb", f"{cat}.txt")

def load_json_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_model_config(model_type, config_path="configs/distributed_model_config.json"):
    config = load_json_config(config_path)
    return config.get(model_type, {})

def get_data_loader_config(config_path="configs/data_loader.json"):
    return load_json_config(config_path)

def get_training_config(config_path="configs/training_config.json"):
    return load_json_config(config_path)

def get_cv_config(config_path="configs/cv_test_config.json"):
    return load_json_config(config_path)

def get_callbacks_config(config_path="configs/callbacks.json"):
    return load_json_config(config_path)