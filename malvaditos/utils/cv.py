import os, json
import pandas as pd
from sklearn.model_selection import KFold

from .config_loader import load_hparams
from ..trainer import train_one_from_splits
from .constant import load_callbacks_config

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _split_by_doc(df, train_docs, val_docs, test_docs):
    tr = df[df["document_ID"].astype(str).isin(set(map(str, train_docs)))]
    va = df[df["document_ID"].astype(str).isin(set(map(str, val_docs)))]
    te = df[df["document_ID"].astype(str).isin(set(map(str, test_docs)))]
    return tr, va, te

def _carve_val(train_docs, frac = 0.2):
    if not train_docs: return [], []
    n_val = max(1, int(len(train_docs)*frac))
    s = sorted(map(str, train_docs))
    return [d for d in s[n_val:]], s[:n_val]

def run_cross_validation(
    csv_path,
    distributed_cfg,
    out_dir,
    k = 5,
    prebuilt_folds_json = None,
    seed = 42,
    max_len = 128,
    batch_size = 32,
):
    hparams = load_hparams(distributed_cfg)
    callbacks_cfg = load_callbacks_config()
    df = pd.read_csv(csv_path)
    if "document_ID" not in df.columns:
        raise ValueError("CSV debe contener columna 'document_ID'.")

    folds = []
    if prebuilt_folds_json and os.path.exists(prebuilt_folds_json):
        pre = json.load(open(prebuilt_folds_json, "r", encoding="utf-8"))
        for k_str in sorted(pre.keys(), key=lambda x: int(x)):
            folds.append((list(map(str, pre[k_str]["train"])), list(map(str, pre[k_str]["test"]))))
    else:
        doc_ids = sorted(df["document_ID"].astype(str).unique().tolist())
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(doc_ids):
            tr_docs = [doc_ids[i] for i in tr_idx]
            te_docs = [doc_ids[i] for i in te_idx]
            folds.append((tr_docs, te_docs))

    _ensure_dir(out_dir)
    ckpts, per_fold_metrics = [], []
    for i, (train_docs, test_docs) in enumerate(folds, start=1):
        train_docs_final, val_docs = _carve_val(train_docs, 0.2)
        fold_dir = os.path.join(out_dir, f"fold_{i}")
        _ensure_dir(fold_dir)
        tr_df, va_df, te_df = _split_by_doc(df, train_docs_final, val_docs, test_docs)
        tr_csv, va_csv, te_csv = os.path.join(fold_dir, "train.csv"), os.path.join(fold_dir, "val.csv"), os.path.join(fold_dir, "test.csv")
        tr_df.to_csv(tr_csv, index=False); va_df.to_csv(va_csv, index=False); te_df.to_csv(te_csv, index=False)

        ckpt, metrics = train_one_from_splits(
            train_csv=tr_csv, val_csv=va_csv, test_csv=te_csv,
            kb_dir_unused="local_database/KB",
            hparams=hparams,
            out_dir=fold_dir,
            max_len=max_len,
            batch_size=batch_size,
            callbacks_config=callbacks_cfg,
        )
        ckpts.append(ckpt)
        per_fold_metrics.append({"fold": i, **metrics})

    def _avg(key: str):
        vals = [m.get(key) for m in per_fold_metrics if m.get(key) is not None]
        return sum(vals)/len(vals) if vals else None

    summary = {
        "fold_checkpoints": ckpts,
        "per_fold_metrics": per_fold_metrics,
        "avg_val_f1": _avg("val_f1"),
        "avg_test_f1": _avg("test_f1"),
        "avg_test_accuracy": _avg("test_accuracy"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary