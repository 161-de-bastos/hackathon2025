import os
import pandas as pd
import matplotlib.pyplot as plt

def _find_metrics_csv(run_dir):
    logs = os.path.join(run_dir, "logs")
    if not os.path.isdir(logs):
        raise FileNotFoundError(f"No 'logs' dir in {run_dir}")
    versions = sorted([d for d in os.listdir(logs) if d.startswith("version_")])
    if not versions:
        raise FileNotFoundError(f"No versions in {logs}")
    csv_path = os.path.join(logs, versions[-1], "metrics.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics.csv not found at {csv_path}")
    return csv_path

def _smooth(y, k=1):
    if k <= 1: 
        return y
    import numpy as np
    y = pd.Series(y).rolling(k, min_periods=1).mean().values
    return y

def plot_losses(run_dir, smooth=1, show=True, save_path=None):
    csv_path = _find_metrics_csv(run_dir)
    df = pd.read_csv(csv_path)
    df_epoch = df[df["step"].isna()]
    tr = df_epoch[df_epoch["metric"] == "train_loss"]
    va = df_epoch[df_epoch["metric"] == "val_loss"]
    plt.figure()
    if not tr.empty:
        plt.plot(tr["epoch"], _smooth(tr["value"].values, smooth), label="train_loss")
    if not va.empty:
        plt.plot(va["epoch"], _smooth(va["value"].values, smooth), label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Train/Val Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

def plot_f1(run_dir, smooth=1, show=True, save_path=None):
    csv_path = _find_metrics_csv(run_dir)
    df = pd.read_csv(csv_path)
    df_epoch = df[df["step"].isna()]
    trf = df_epoch[df_epoch["metric"] == "train_f1_multi"]
    vf  = df_epoch[df_epoch["metric"] == "val_f1_score"]
    plt.figure()
    if not trf.empty:
        plt.plot(trf["epoch"], _smooth(trf["value"].values, smooth), label="train_f1_multi")
    if not vf.empty:
        plt.plot(vf["epoch"], _smooth(vf["value"].values, smooth), label="val_f1_score")
    plt.xlabel("epoch"); plt.ylabel("F1 (macro over 5 cats)")
    plt.title("Train/Val F1")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()