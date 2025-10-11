import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score

from .token import tokenizer_export_state, tokenizer_from_state
from .model import Memory

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

        self.f1_metric = BinaryF1Score()
        self._tokenizer_state = None

    def forward(self, x):
        return self.model(x, self.kb_ids, self.kb_mask)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _strong_hinge(self, att_weights, strong_batch):
        if not self.ps_flag or att_weights.numel() == 0:
            return att_weights.sum() * 0.0
        B, M = att_weights.shape
        losses = []
        for b in range(B):
            pos = []
            for _, lst in strong_batch[b].items():
                pos = lst; break
            if not pos:
                continue
            pos = [i for i in pos if 0 <= i < M]
            if not pos:
                continue
            w = att_weights[b]                                  # [M]
            pos_mask = torch.zeros(M, dtype=torch.bool, device=w.device); pos_mask[pos] = True
            neg_mask = ~pos_mask
            if not neg_mask.any():
                continue
            w_pos = w[pos_mask][:, None]                        # [P,1]
            w_neg = w[neg_mask][None, :]                        # [1,N]
            margin = self.ps_margin - w_pos + w_neg
            losses.append(margin.clamp_min(0).mean())
        if not losses:
            return att_weights.sum() * 0.0
        return torch.stack(losses).mean() * self.ps_coeff

    def _step(self, batch, stage: str):
        x = batch["input_ids"]          # [B,T]
        y = batch["labels"].float()     # [B]
        logits, att, _ = self(x)        # logits:[B], att:[B,M]
        cls = F.binary_cross_entropy_with_logits(logits, y)
        loss = cls + self._strong_hinge(att, batch["strong"]) if self.ps_flag else cls

        preds = (logits.sigmoid() > 0.5).to(torch.int)
        f1 = self.f1_metric(preds, y.to(torch.int))

        if stage == "val":
            self.log("val_f1_score", f1, prog_bar=True, on_epoch=True, on_step=False)
        elif stage == "test":
            self.log("test_f1", f1, prog_bar=True, on_epoch=True, on_step=False)

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