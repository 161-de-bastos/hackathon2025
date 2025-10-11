import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score, BinaryF1Score

from .model import Memory
from .token import tokenizer_export_state, tokenizer_from_state
from .utils.constant import KB_CATEGORIES

class LightMemory(pl.LightningModule):
    def __init__(self, 
            vocab_size, 
            pad_idx, 
            hparams, 
            kb_ids, 
            kb_mask     
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["kb_ids", "kb_mask"])
        self.model = Memory(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            embed_dim=hparams.get("embedding_dim", hparams.get("embed_dim", 100)),
            hidden_dim=hparams.get("hidden_dim", 100),
            dropout=hparams.get("dropout", 0.1),
        )
        self.register_buffer("kb_ids", kb_ids.long(), persistent=False)
        self.register_buffer("kb_mask", kb_mask.bool(), persistent=False)

        ps = hparams.get("partial_supervision_info", {}).get("value", {})
        self.ps_flag   = bool(ps.get("flag", False))
        self.ps_coeff  = float(ps.get("coefficient", 1.0))
        self.ps_margin = float(ps.get("margin", 0.5))

        self.lr = float(hparams.get("learning_rate", 2e-3))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))

        self.f1_multi = MultilabelF1Score(num_labels=len(KB_CATEGORIES), average="macro", threshold=0.5)
        self.f1_general = BinaryF1Score()
        self._tokenizer_state = None

    def forward(self, x):
        return self.model(x, self.kb_ids, self.kb_mask)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _strong_hinge(self, att_dict, strong_batch):
        if not self.ps_flag:
            any_att = next(iter(att_dict.values()))
            return any_att.sum() * 0.0
        
        losses = []
        for cat, w in att_dict.items():
            B, M = w.shape
            for b in range(B):
                pos = list(strong_batch[b].get(cat, []))
                if not pos:
                    continue
                pos = [i for i in pos if 0 <= i < M]
                if not pos:
                    continue
                mask_pos = torch.zeros(M, dtype=torch.bool, device=w.device); mask_pos[pos] = True
                mask_neg = ~mask_pos
                if not mask_neg.any():
                    continue
                w_pos = w[b][mask_pos][:, None]
                w_neg = w[b][mask_neg][None, :]
                margin = self.ps_margin - w_pos + w_neg
                losses.append(margin.clamp_min(0).mean())
        if not losses:
            any_att = next(iter(att_dict.values()))
            return any_att.sum() * 0.0
        return torch.stack(losses).mean() * self.ps_coeff
    
    def _step(self, batch, stage):
        x = batch["input_ids"]
        y_multi = batch["labels_multi"]
        y_gen   = batch["label_general"]
        logits_multi, att, _ = self(x)  

        loss_cls = F.binary_cross_entropy_with_logits(logits_multi, y_multi)
        if self.ps_flag:
            loss = loss_cls + self._strong_hinge(att, batch["strong"])
        else:
            loss = loss_cls

        probs_multi = torch.sigmoid(logits_multi)
        preds_multi = (probs_multi > 0.5).float()
        preds_gen   = (preds_multi.max(dim=1).values > 0.5).float()

        f1m = self.f1_multi(preds_multi, y_multi.to(torch.int))
        f1g = self.f1_general(preds_gen.to(torch.int), y_gen.to(torch.int))

        if stage == "train":
            self.log("train_f1_score", f1m, prog_bar=False, on_epoch=True, on_step=False)
            self.log("train_f1_general", f1g, prog_bar=False, on_epoch=True, on_step=False)
        if stage == "val":
            self.log("val_f1_score", f1m, prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_f1_general", f1g, prog_bar=False, on_epoch=True, on_step=False)
        elif stage == "test":
            self.log("test_f1_multi", f1m, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_f1_general", f1g, prog_bar=True, on_epoch=True, on_step=False)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def attach_tokenizer(self, tok):
        self._tokenizer_state = tokenizer_export_state(tok)

    def on_save_checkpoint(self, checkpoint):
        if self._tokenizer_state is not None:
            checkpoint["tokenizer_state"] = self._tokenizer_state

    def on_load_checkpoint(self, checkpoint):
        self._tokenizer_state = checkpoint.get("tokenizer_state", None)

    def get_tokenizer(self):
        return tokenizer_from_state(self._tokenizer_state) if self._tokenizer_state else None