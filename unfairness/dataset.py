import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .token import pad_sequences
from .constant import KB_CATEGORIES, kb_file_for

class ToS(Dataset):
    def __init__(self, df, category, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.cat = category
        self.tok = tokenizer
        self.max_len = max_len

        seqs = self.tok.texts_to_sequences(self.df["text"].astype(str).tolist())
        maxlen = max_len if max_len > 0 else (max((len(s) for s in seqs), default=1))
        self.seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post", value=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        label = float(row[self.cat]) if self.cat in row else 0.0
        strong = {}
        tk = f"{self.cat}_targets"
        if self.cat in KB_CATEGORIES and tk in row and isinstance(row[tk], str) and row[tk].strip().startswith("["):
            try:
                strong[self.cat] = list(json.loads(row[tk]))
            except Exception:
                strong[self.cat] = []
        else:
            strong[self.cat] = []
        return {
            "input_ids": torch.tensor(self.seqs[i], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "strong": strong,
        }

def load_kb_bank(category, tokenizer, max_len):
    if category not in KB_CATEGORIES:
        return torch.zeros((0, max_len), dtype=torch.long), torch.zeros((0,), dtype=torch.bool)

    path = kb_file_for(category)
    lines = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    if not lines:
        lines = [""]  # dummy para shape estable

    ids = tokenizer.texts_to_sequences(lines)
    ids = pad_sequences(ids, maxlen=max_len, padding="post", truncating="post", value=0)
    kb_ids = torch.tensor(ids, dtype=torch.long)
    kb_mask = torch.ones(kb_ids.size(0), dtype=torch.bool)
    return kb_ids, kb_mask

def make_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    def collate(batch):
        x = torch.stack([b["input_ids"] for b in batch], 0)  # [B,T]
        y = torch.stack([b["label"] for b in batch], 0)      # [B]
        strong = [b["strong"] for b in batch]                # len B
        return {"input_ids": x, "labels": y, "strong": strong}
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate) if val_ds else None
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate) if test_ds else None
    return tr, va, te
