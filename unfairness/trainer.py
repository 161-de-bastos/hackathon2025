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