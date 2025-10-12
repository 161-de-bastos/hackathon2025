import pandas as pd
import matplotlib.pyplot as plt

def _read_metrics_wide(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["epoch","step","train_loss","val_loss","train_f1_score","val_f1_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "step" in df.columns:
        df = df.sort_values(["epoch","step"])
    else:
        df = df.sort_values(["epoch"])
    df_last = df.groupby("epoch", as_index=False).last()
    return df_last

def _smooth_vec(vals, k=1):
    if not k or k <= 1:
        return vals
    return pd.Series(vals).rolling(k, min_periods=1).mean().values

def plot_losses(csv_path, smooth=1, show=True, save_path=None):
    df = _read_metrics_wide(csv_path)
    plt.figure()
    any_line = False
    if "train_loss" in df.columns and df["train_loss"].notna().any():
        plt.plot(df["epoch"], _smooth_vec(df["train_loss"].values, smooth), label="train_loss")
        any_line = True
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        plt.plot(df["epoch"], _smooth_vec(df["val_loss"].values, smooth), label="val_loss")
        any_line = True
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train / Val Loss")
    if any_line: plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches="tight")
    if show: plt.show()

def plot_f1(csv_path, smooth=1, show=True, save_path=None):
    df = _read_metrics_wide(csv_path)
    plt.figure()
    any_line = False
    if "train_f1_score" in df.columns and df["train_f1_score"].notna().any():
        plt.plot(df["epoch"], _smooth_vec(df["train_f1_score"].values, smooth), label="train_f1_score")
        any_line = True
    if "val_f1_score" in df.columns and df["val_f1_score"].notna().any():
        plt.plot(df["epoch"], _smooth_vec(df["val_f1_score"].values, smooth), label="val_f1_score")
        any_line = True
    plt.xlabel("epoch")
    plt.ylabel("F1 (macro)")
    plt.title("Train / Val F1")
    if any_line: plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches="tight")
    if show: plt.show()