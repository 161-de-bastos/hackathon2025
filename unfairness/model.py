import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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

class LightningMemory(pl.LightningModule):
    def __init__(self, 
            vocab_size,
            embed_dim,
            memory_vects,
            learning_rate = 1e-3,
            partial_sup = None
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["memory_vectors"])
        self.model = Memory(vocab_size, embed_dim, memory_vects)
        self.learning_rate = learning_rate

        if partial_sup is None:
            partial_sup = {"flag": False}
        self.strong_sup_flag = partial_sup.get("flag", False)
        self.strong_sup_coeff = partial_sup.get("coefficient", 1.0)
        self.margin = partial_sup.get("margin", 0.5)

        from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
        self.val_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()

    def forward(self, query_seq):
        return self.model(query_seq)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_size):
        seq, label, target_list = batch
        logits, weights_list, scores_list = [], [], []

        for i in range(seq.size(0)):
            q_seq = seq[i]
            logit, weights, scores = self.model(q_seq)
            logits.append(logit)
            weights_list.append(weights)
            scores_list.append(scores)

        logits = torch.stack(logits).squeeze(1)
        labels = label.float()
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        ss_loss = 0.0
        if self.strong_sup_flag:
            batch_ss_losses = []
            for i in range(len(target_list)):
                true_indices = target_list[i]
                if len(true_indices) == 0:
                    continue

                w = torch.sigmoid(scores_list[i])
                if w.numel() == 0:
                    continue

                M = w.size(0)
                pos_idx = set(true_indices)

                pos_idx = [j for j in pos_idx if 0 <= j < M]
                if not pos_idx:
                    continue
                neg_idx = [j for j in range(M) if j not in pos_idx]
                if not neg_idx:
                    continue

                w_pos = w[list(pos_idx)]
                w_neg = w[neg_idx]

                w_pos_expand = w_pos.view(-1, 1)
                w_neg_expand = w_neg.view(1, -1)

                margin_loss_matrix = self.margin - w_pos_expand + w_neg_expand
                margin_loss_matrix = torch.clamp(margin_loss_matrix, min=0.0)
                loss_per_sample = margin_loss_matrix.mean()
                batch_ss_losses.append(loss_per_sample)
            if batch_ss_losses:
                ss_loss = torch.stack(batch_ss_losses).mean()

        loss = cls_loss + ss_loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_size):
        seq, label, target_list = batch
        logits = []
        for i in range(seq.size(0)):
            logit, _, _ = self.model(seq[i])
            logits.append(logit)
        logits = torch.stack(logits).squeeze(1)
        preds = torch.sigmoid(logits)
        pred_labels = (preds >= 0.5).int()
        self.val_f1(pred_labels, label.int())
        self.val_acc(pred_labels, label.int())
        val_loss = F.binary_cross_entropy_with_logits(logits, label.float())
        self.log("val_loss", val_loss, prog_bar=False, logger=True, on_epoch=True)
        return {}
    
    def on_validation_epoch_end(self):
        val_f1_score = self.val_f1.compute()
        val_acc = self.val_acc.compute()
        
        self.log("val_f1_score", val_f1_score, prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc, prog_bar=True, logger=True)

        self.val_f1.reset()
        self.val_acc.reset()

    def test_step(self, batch):
        seq, label, _ = batch
        logits = []
        for i in range(seq.size(0)):
            logit, _, _ = self.model(seq[i])
            logits.append(logit)
        logits = torch.stack(logits).squeeze(1)
        preds = torch.sigmoid(logits)
        pred_labels = (preds >= 0.5).int()

        self.val_f1(pred_labels, label.int())
        self.val_acc(pred_labels, label.int())
        return {}
        
    def on_test_epoch_end(self):
        test_f1 = self.val_f1.compute()
        test_acc = self.val_acc.compute()
        self.log("test_f1_score", test_f1, prog_bar=True, logger=True)
        self.log("test_accuracy", test_acc, prog_bar=True, logger=True)
        self.val_f1.reset()
        self.val_acc.reset()
