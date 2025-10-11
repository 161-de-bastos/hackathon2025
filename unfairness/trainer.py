import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .token import tokenizer
from .dataset import ToS, load_kb_bank, make_dataloaders
from .lightning import LightMemory

def _build_splits_datasets(train_csv, val_csv, test_csv,
                           category, max_len):
    df_tr = pd.read_csv(train_csv)
    tok = tokenizer(oov_token="<OOV>", lower=True)
    tok.fit_on_texts(df_tr["text"].astype(str).tolist())

    df_va = pd.read_csv(val_csv) if val_csv else None
    df_te = pd.read_csv(test_csv) if test_csv else None

    ds_tr = ToS(df_tr, category, tok, max_len)
    ds_va = ToS(df_va, category, tok, max_len) if df_va is not None else None
    ds_te = ToS(df_te, category, tok, max_len) if df_te is not None else None
    return tok, ds_tr, ds_va, ds_te

def _make_trainer(save_dir, callbacks_config):
    os.makedirs(save_dir, exist_ok=True)
    es_cfg = callbacks_config.get("earlystopping", {
        "monitor": "val_f1_score", "mode": "max", "patience": 20, "min_delta": 0.0
    })
    ckpt = ModelCheckpoint(dirpath=save_dir, filename="best", save_top_k=1,
                           monitor=es_cfg["monitor"], mode=es_cfg["mode"])
    es = EarlyStopping(monitor=es_cfg["monitor"], mode=es_cfg["mode"],
                       patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])
    trainer = pl.Trainer(max_epochs=100, callbacks=[ckpt, es], log_every_n_steps=10, default_root_dir=save_dir)
    return trainer, ckpt

def train_one_from_splits(
    train_csv,
    val_csv,
    test_csv,
    hparams,
    out_dir,
    category,
    max_len = 128,
    batch_size = 32,
    callbacks_config = None
):
    tok, ds_tr, ds_va, ds_te = _build_splits_datasets(train_csv, val_csv, test_csv, category, max_len)
    kb_ids, kb_mask = load_kb_bank(category, tok, max_len)
    tr, va, te = make_dataloaders(ds_tr, ds_va, ds_te, batch_size)

    lit = LightMemory(vocab_size=tok.vocab_size, pad_idx=0, hparams=hparams, kb_ids=kb_ids, kb_mask=kb_mask)
    trainer, ckpt_cb = _make_trainer(out_dir, callbacks_config or {})

    trainer.fit(lit, tr, va)
    best_ckpt = ckpt_cb.best_model_path or os.path.join(out_dir, "best.ckpt")

    val_f1 = float(trainer.callback_metrics.get("val_f1_score", 0.0))
    metrics = {"val_f1": val_f1, "test_f1": None, "test_accuracy": None}
    if te is not None:
        res = trainer.test(lit, dataloaders=te, ckpt_path=best_ckpt)
        if res:
            metrics["test_f1"] = float(res[0].get("test_f1", 0.0))
            metrics["test_accuracy"] = float(res[0].get("test_accuracy", 0.0))
    return best_ckpt, metrics