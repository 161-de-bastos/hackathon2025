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

        texts = self.df["text"].astype(str).tolist()
        seqs = self.tok.texts_to_sequences(texts)
        maxlen = max_len if max_len > 0 else (max((len(s) for s in seqs), default=1))
        self.seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post", value=0)
        self._raw_texts = texts

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
            "raw_text": self._raw_texts[i]
        }

def _vec_texts(tokenizer, texts, max_len):
    ids = tokenizer.texts_to_sequences(texts)
    ids = pad_sequences(ids, maxlen=max_len, padding="post", truncating="post", value=0)
    return torch.tensor(ids, dtype=torch.long)

def make_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    def collate(batch):
        x = torch.stack([b["input_ids"] for b in batch], 0)
        y_multi = torch.stack([b["labels_multi"] for b in batch], 0)
        y_gen = torch.stack([b["label_general"] for b in batch], 0)
        strong = [b["strong"] for b in batch]
        raw_texts = [b.get("raw_text", None) for b in batch] 
        return {"input_ids": x, "labels_multi": y_multi, "label_general": y_gen, "strong": strong, "raw_text": raw_texts}
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate) if val_ds else None
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate) if test_ds else None
    return tr, va, te


def load_kb_bank(tokenizer, max_len, kb_dir):
    struct = load_kb_struct(kb_dir)
    texts_dict = kb_texts_only(struct)
    kb_ids, kb_mask = {}, {}

    for cat in KB_CATEGORIES:
        ts = texts_dict[cat] if texts_dict[cat] else [""]
        ids = _vec_texts(tokenizer, ts, max_len)
        kb_ids[cat] = ids
        kb_mask[cat] = torch.ones(ids.size(0), dtype=torch.bool)
    return kb_ids, kb_mask

def _csv_path_for(cat, kb_dir):
    return os.path.join(kb_dir, f"{cat}_explanations.csv")

def load_kb_struct(kb_dir="local_database/KB"):
    out = {}
    for cat in KB_CATEGORIES:
        p = _csv_path_for(cat, kb_dir)
        texts, ids, tags = [], [], []
        if os.path.isfile(p):
            df = pd.read_csv(p)
            col_text = "Explanation" if "Explanation" in df.columns else (df.columns[0] if len(df.columns) else None)
            for _, r in df.iterrows():
                t = str(r.get(col_text, "")).strip()
                if not t: 
                    continue
                texts.append(t)
                ids.append(r.get("Id", None))
                tags.append(r.get("Tag", None))
        if not texts:
            texts, ids, tags = [""], [None], [None]
        out[cat] = {"text": texts, "id": ids, "tag": tags}
    return out


def kb_texts_only(kb_struct):
    return {cat: kb_struct[cat]["text"] for cat in kb_struct}
