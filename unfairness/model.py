import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Memory(nn.Module):
    def __init__(self, 
            vocab_size,
            pad_idx,
            embed_dim = 100,
            hidden_dim = 100,
            dropout = 0.1     
        ):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.sim_fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.sim_fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(embed_dim, 1)

    def _masked_mean(self, emb, ids):
        """
        emb: [*, T, E] ; ids: [*, T] con padding_idx=0.
        Retorna: [*, E] media sólo sobre tokens != pad_idx.
        """
        mask = (ids != self.pad_idx).float().unsqueeze(-1)  # [*, T, 1]
        summed = (emb * mask).sum(dim=1)                    # [*, E]
        denom = mask.sum(dim=1).clamp_min(1.0)              # [*, 1]
        return summed / denom

    def _encode_sentences(self, ids):
        # ids: [B,T] o [M,T] → [B,E] o [M,E]
        e = self.emb(ids)                    # [*,T,E]
        return self._masked_mean(e, ids)     # [*,E]

    def forward(self, x, kb_ids, kb_mask):
        B = x.size(0)
        q = self._encode_sentences(x)  # [B,E]

        if kb_ids.numel() == 0 or kb_ids.size(0) == 0:
            w = x.new_zeros((B, 0), dtype=torch.float)
            c = x.new_zeros((B, q.size(1)), dtype=torch.float)
            s = w
        else:
            m = self._encode_sentences(kb_ids)          # [M,E]
            M = m.size(0)
            q_expand = q.unsqueeze(1).expand(B, M, -1)  # [B,M,E]
            m_expand = m.unsqueeze(0).expand(B, M, -1)  # [B,M,E]
            z = torch.cat([q_expand, m_expand], dim=-1) # [B,M,2E]
            h = F.relu(self.sim_fc1(z))                 # [B,M,E]
            s = self.sim_fc2(self.drop(h)).squeeze(-1)  # [B,M] (raw scores)
            mask = kb_mask.to(x.device).unsqueeze(0).expand_as(s)  # [B,M]
            s = s.masked_fill(~mask, float("-inf"))
            w = torch.sigmoid(s) * mask.float()         # [B,M] independiente, sin normalizar
            c = (w.unsqueeze(-1) * m_expand).sum(dim=1) # [B,E]

        out = self.drop(q + c)              # [B,E]
        logit = self.cls(out).squeeze(-1)   # [B]
        return logit, w, s