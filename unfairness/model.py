import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.constant import KB_CATEGORIES

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    def forward(self, ids):
        e = self.emb(ids)                              # [*,T,E]
        mask = (ids != self.pad_idx).float().unsqueeze(-1)
        summed = (e * mask).sum(1)
        denom = mask.sum(1).clamp_min(1.0)
        return summed / denom
    
class Memory(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed_dim=100, hidden_dim=100, dropout=0.1):
        super().__init__()
        self.encoder = SentenceEncoder(vocab_size, embed_dim, pad_idx)
        self.sim_fc1 = nn.ModuleDict({cat: nn.Linear(2*embed_dim, hidden_dim) for cat in KB_CATEGORIES})
        self.sim_fc2 = nn.ModuleDict({cat: nn.Linear(hidden_dim, 1)           for cat in KB_CATEGORIES})
        self.cls     = nn.ModuleDict({cat: nn.Linear(embed_dim, 1)            for cat in KB_CATEGORIES})
        self.drop = nn.Dropout(dropout)

    def _scores_for_cat(self, cat, q, m, mask):
        B, E = q.shape
        M = m.size(0)
        qx = q.unsqueeze(1).expand(B, M, E)
        mx = m.unsqueeze(0).expand(B, M, E)
        z  = torch.cat([qx, mx], dim=-1)
        h  = F.relu(self.sim_fc1[cat](z))
        s  = self.sim_fc2[cat](self.drop(h)).squeeze(-1)    
        s  = s.masked_fill(~mask, float("-inf"))
        w  = torch.sigmoid(s) * mask.float()                
        return s, w

    def forward(self, x, kb_ids, kb_mask):
        B = x.size(0)
        q = self.encoder(x)                         
        logits = []
        att_out = {}
        scores_out = {}
        for cat in KB_CATEGORIES:
            ids = kb_ids[cat]                       
            m = self.encoder(ids)                   
            mask = kb_mask[cat].to(x.device).unsqueeze(0).expand(B, -1)
            s, w = self._scores_for_cat(cat, q, m, mask)
            c = (w.unsqueeze(-1) * m.unsqueeze(0)).sum(1)   
            out = self.drop(q + c)
            logit = self.cls[cat](out).squeeze(-1)          
            logits.append(logit)
            att_out[cat] = w
            scores_out[cat] = s
        logits = torch.stack(logits, dim=1)  # [B,5]
        return logits, att_out, scores_out