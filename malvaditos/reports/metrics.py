import json
import numpy as np
from sklearn.metrics import classification_report

def multilabel_report(y_true, y_pred, target_names=None, output_dict=True):
    rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=output_dict, zero_division=0)
    return rep

def general_report(y_true, y_pred, output_dict=True):
    rep = classification_report(y_true, y_pred, target_names=["general"], output_dict=output_dict, zero_division=0)
    return rep

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)