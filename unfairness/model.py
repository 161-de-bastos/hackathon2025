import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            memory_vects
        ):
        super(Memory).__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if memory_vects is not None:
            self.register_buffer('memory_vectors',memory_vects)
        else:
            self.memory_vectors = None

        self.similarity_fc1 = nn.Linear(2 * embed_dim, embed_dim)
        self.similarity_fc2 = nn.Linear(embed_dim, 1)

        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, query_seq):
        q_word_embeds = self.embedding(query_seq)                   # (seq_len, embed_dim)
        q_vec = torch.mean(q_word_embeds, dim = 0, keepdim = True)  # (1, embed_dom)

        if self.memory_vectors is not None:
            M = self.memory_vectors.shape[0]
            mem_vecs = self.memory_vectors                          # (M, embed_dim)
        else:
            M = 0
            mem_vecs = torch.zeros((0, self.embed_dim), device = q_vec.device)

        if M > 0:
            q_expand = q_vec.expand(M, -1)                          # (M, embed_dim)
            q_concat = torch.cat([q_expand, mem_vecs], dim = 1)     # (M, 2*embed_dim)
            hidden = F.relu(self.similarity_fc1(q_concat))          # (M, embed_dim)
            scores = self.similarity_fc2(hidden).squeeze(1)         # (M)
        else:
            scores = torch.zeros((0,), device=q_vec.device)         # (M)

        w1 = F.softmax(scores, dim = 0) if M > 0 else scores

        if M > 0:
            w2 = w1.unsqueeze(1)                                    # (M, 1)
            c_vec = torch.sum(w2 * mem_vecs, dim=0, keepdim=True)   # (1, embed_dim)
        else:
            c_vec = torch.zeros_like(q_vec)                         # (1, embed_dim)

        out_vec = q_vec + c_vec                                     # (1, embed_dim)
        logits = self.classifier(out_vec).squeeze(1)
        return logits, w1, scores






