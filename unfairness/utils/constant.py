import json
import os

KB_CATEGORIES = ("A", "CH", "CR", "LTD", "TER")

def kb_file_for(cat):
    return os.path.join("local_database","KB", f"{cat}.txt")

def load_json_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_callbacks_config(path = "configs/callbacks.json"):
    return load_json(path)

def load_training_config(path = "configs/training_config.json"):
    return load_json(path)

def load_data_loader_config(path = "configs/data_loader.json"):
    return load_json(path)

def load_distributed_model_config(path = "configs/distributed_model_config.json"):
    return load_json(path)