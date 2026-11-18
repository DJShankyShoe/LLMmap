import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
from functools import partial
from typing import Dict, Any
from torch.optim import Optimizer

from .inference_model_archs import InferenceModelLLMmap, make_siamese_network


OPTIMIZERS: Dict[str, torch.optim.Optimizer] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD":  torch.optim.SGD,
}

class LLMmapTrainerClosed(pl.LightningModule):
    def __init__(self, model, hparams):
        """
        net      – an instance of InferenceModelLLMmap
        hparams  – the same dict that holds optimizer specs, num_classes, …
        """
        super().__init__()
        self.net        = model
        self.save_hyperparameters(hparams)      # logs LR, heads, etc.

        self.criterion  = torch.nn.CrossEntropyLoss()
        self.train_acc  = Accuracy(task="multiclass",
                                   num_classes=hparams["num_classes"])
        self.val_acc    = Accuracy(task="multiclass",
                                   num_classes=hparams["num_classes"])

    # ----- forward pass -------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # ----- training -----------------------------------------------------
    def training_step(self, batch, _):
        x, y     = batch
        logits   = self(x)
        loss     = self.criterion(logits, y)

        self.train_acc.update(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ----- validation (= test every epoch) ------------------------------
    def validation_step(self, batch, _):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)

        self.val_acc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc",  self.val_acc,
                 on_step=False, on_epoch=True, prog_bar=True)

    # reset metric states each epoch
    def on_train_epoch_start(self):     self.train_acc.reset()
    def on_validation_epoch_start(self): self.val_acc.reset()

    # ----- optimiser ----------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        """
        Returns a torch.optim.* instance, accepting either
        1) the new JSON-friendly dict  {"name": <str>, "params": {...}}
        2) the legacy tuple            (opt_cls, kwargs)
        """
        opt_cfg = self.hparams["optimizer"]
    
        # ── new dict style ───────────────────────────────────────────────
        if isinstance(opt_cfg, dict):
            name   = opt_cfg["name"]
            kwargs = opt_cfg.get("params", {})
            try:
                opt_cls = OPTIMIZERS[name]
            except KeyError:                     # unknown name -> clear error
                raise ValueError(
                    f"Unknown optimizer '{name}'. "
                    f"Available: {list(OPTIMIZERS)}"
                )
            return opt_cls(self.parameters(), **kwargs)
    
        opt_cls, opt_kw = opt_cfg
        return opt_cls(self.parameters(), **opt_kw)




class ContrastiveLoss(nn.Module):
    """
    Classic contrastive loss for Siamese networks.

    Args
    ----
    margin : float, default=1.0
        Distance margin that separates positive and negative pairs.

    Shape
    -----
    y_pred : (B, 1)  – model output in [0, 1] (after sigmoid)
    y_true : (B, 1)  – binary label: 0 = "same" / positive pair,
                       1 = "different" / negative pair
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ensure type/shape consistency
        y_true = y_true.float().view_as(y_pred)

        square_pred   = y_pred.pow(2)
        margin_square = (torch.clamp(self.margin - y_pred, min=0.0)).pow(2)

        loss = ((1.0 - y_true) * square_pred + y_true * margin_square).mean()
        return loss

class LLMmapTrainerSiamese(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: dict):
        """
        model    – the SiameseNetwork instance that ends with a sigmoid.
        hparams  – same dict you already use (must include "optimizer").
        """
        super().__init__()
        self.net = model
        self.save_hyperparameters(hparams)

        self.criterion = ContrastiveLoss(margin=hparams.get("margin", 1.0))

        # Binary metrics (0 = similar, 1 = dissimilar)
        self.train_acc = BinaryAccuracy()
        self.val_acc   = BinaryAccuracy()

    # ----- forward pass -------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # ----- training -----------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch                          # y ∈ {0,1}
        y_hat = self(x)[:, 0]                       # (B,1) in [0,1]
        loss  = self.criterion(y_hat, y)

        self.train_acc.update(y_hat, y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  self.train_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ----- validation ---------------------------------------------------
    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)[:, 0]
        loss  = self.criterion(y_hat, y)

        self.val_acc.update(y_hat, y.int())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc",  self.val_acc,
                 on_step=False, on_epoch=True, prog_bar=True)

    # reset metric states each epoch
    def on_train_epoch_start(self):     self.train_acc.reset()
    def on_validation_epoch_start(self): self.val_acc.reset()

    def configure_optimizers(self) -> Optimizer:
        """
        Returns a torch.optim.* instance, accepting either
        1) the new JSON-friendly dict  {"name": <str>, "params": {...}}
        2) the legacy tuple            (opt_cls, kwargs)
        """
        opt_cfg = self.hparams["optimizer"]
    
        # ── new dict style ───────────────────────────────────────────────
        if isinstance(opt_cfg, dict):
            name   = opt_cfg["name"]
            kwargs = opt_cfg.get("params", {})
            try:
                opt_cls = OPTIMIZERS[name]
            except KeyError:                     # unknown name -> clear error
                raise ValueError(
                    f"Unknown optimizer '{name}'. "
                    f"Available: {list(OPTIMIZERS)}"
                )
            return opt_cls(self.parameters(), **kwargs)
    
        # ── old tuple style (cls, kwargs) ────────────────────────────────
        opt_cls, opt_kw = opt_cfg
        return opt_cls(self.parameters(), **opt_kw)


def _make_optimizer(params, opt_cfg):
    if isinstance(opt_cfg, dict):
        return OPTIMIZERS[opt_cfg["name"]](params, **opt_cfg.get("params", {}))
    opt_cls, opt_kw = opt_cfg
    return opt_cls(params, **opt_kw)


def batch_hard_triplet_loss(feats, labels, margin=0.3, normalize=True):
    """Batch-hard triplet loss (Hermans et al., 2017).

    For every anchor: hardest positive (farthest same-LLM trace) vs hardest negative
    (closest other-LLM trace). Optimises exactly what open-set nearest-template matching needs.
    Returns (loss, fraction of anchors whose hardest negative is farther than the hardest positive).
    """
    if normalize:
        feats = F.normalize(feats, dim=-1)
    d = torch.cdist(feats, feats)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=feats.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    hardest_pos = (d * pos_mask).max(dim=1).values
    hardest_neg = d.masked_fill(~neg_mask, float('inf')).min(dim=1).values
    valid = pos_mask.any(1) & neg_mask.any(1)
    loss = F.relu(hardest_pos - hardest_neg + margin)[valid].mean()
    acc = (hardest_neg > hardest_pos)[valid].float().mean()
    return loss, acc


class LLMmapTrainerTriplet(pl.LightningModule):
    """Open-set training with PK batches + batch-hard triplet loss on the L2-normalised features."""

    def __init__(self, model: nn.Module, hparams: dict):
        super().__init__()
        self.net = model
        self.save_hyperparameters(hparams, ignore=['model'])
        self.margin = hparams.get('margin', 0.3)

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage):
        x, y = batch
        loss, acc = batch_hard_triplet_loss(self(x), y, self.margin)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'))
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, 'train')

    def validation_step(self, batch, _):
        self._step(batch, 'val')

    def configure_optimizers(self):
        opt = _make_optimizer(self.parameters(), self.hparams["optimizer"])
        return opt


def train_model(output_dir, siamese, loader_train, loader_test, conf):
    hp = conf['inference_model']
    triplet = siamese and conf.get('open_set_loss', 'contrastive') == 'triplet'

    if triplet:
        inference_model = InferenceModelLLMmap(hp, is_for_siamese=True)
        model = inference_model
        litmod = LLMmapTrainerTriplet(model, hp)
    elif siamese:
        model, inference_model = make_siamese_network(hp)
        litmod = LLMmapTrainerSiamese(model, hp)
    else:
        model = InferenceModelLLMmap(hp)
        inference_model = model
        litmod = LLMmapTrainerClosed(model, hp)

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=conf['training']['early_stop_patience'],
        verbose=True
    )

    ckpt_best = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=conf['training']['max_epochs'],
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, ckpt_best],
        log_every_n_steps=conf['training']['log_every_n_steps'],
        gradient_clip_val=conf['training'].get('gradient_clip_val'),
    )

    trainer.fit(litmod, loader_train, loader_test)

    # Restore the best weights. (The original code called load_from_checkpoint with an
    # `hp=` kwarg the LightningModules don't accept; loading the state dict directly is
    # simpler and works on CPU-only machines for GPU-trained checkpoints.)
    best_model_path = ckpt_best.best_model_path
    if best_model_path:
        state = torch.load(best_model_path, map_location='cpu', weights_only=False)['state_dict']
        litmod.load_state_dict(state)

    return litmod, inference_model
