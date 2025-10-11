import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .token import pad_sequences
from .utils.constant import KB_CATEGORIES, kb_file_for

class ToS(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

        seqs = self.tok.texts_to_sequences(self.df["text"].astype(str).tolist())
        maxlen = max_len if max_len > 0 else (max((len(s) for s in seqs), default=1))
        self.seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post", value=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        y_multi = []
        strong = {}

        for cat in KB_CATEGORIES:
            y = float(row.get(cat, 0.0))
            y_multi.append(y)
            tk = f"{cat}_targets"
            if isinstance(row.get(tk), str) and row[tk].strip().startswith("["):
                try:
                    strong[cat] = list(json.loads(row[tk]))
                except Exception:
                    strong[cat] = []
            else:
                strong[cat] = []

        if "label" in row:
            y_general = float(row["label"])
        else:
            y_general = 1.0 if any(v >= 0.5 for v in y_multi) else 0.0
        return {
            "input_ids": torch.tensor(self.seqs[i], dtype=torch.long),
            "labels_multi": torch.tensor(y_multi, dtype=torch.float),
            "label_general": torch.tensor(y_general, dtype=torch.float),
            "strong": strong,
        }

def load_kb_bank(tokenizer, max_len):
    kb_ids, kb_mask = {}, {}
    for cat in KB_CATEGORIES:
        p = kb_file_for(cat)
        lines = []
        if os.path.exists(p):
            with open(p,'r',encoding = "utf-8") as f:
                for ln in f.readlines():
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
        if not lines:
            lines = [""]

        ids = tokenizer.texts_to_sequences(lines)
        ids = pad_sequences(ids, maxlen=max_len, padding="post", truncating="post", value=0)
        kb_ids[cat] = torch.tensor(ids, dtype=torch.long)
        kb_mask[cat] = torch.ones(kb_ids.size(0), dtype=torch.bool)
    return kb_ids, kb_mask

def load_kb_texts(category):
    p = kb_file_for(category)
    if not os.path.exists(p):
        return []
    lines = []
    if os.path.exists(p):
        with open(p,'r',encoding = "utf-8") as f:
            for ln in f.readlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines

def make_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    def collate(batch):
        x = torch.stack([b["input_ids"] for b in batch], 0)
        y_multi = torch.stack([b["labels_multi"] for b in batch], 0)
        y_gen = torch.stack([b["label_general"] for b in batch], 0)
        strong = [b["strong"] for b in batch]
        return {"input_ids": x, "labels_multi": y_multi, "label_general": y_gen, "strong": strong}
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate) if val_ds else None
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate) if test_ds else None
    return tr, va, te
