"""Ultralytics ``ModelEMA``：decay 斜坡 + val/test 用影子权重。

``fit`` 里的 ``ModelCheckpoint`` 若在 ``on_validation_end`` 存盘，此时模型仍是 EMA
（本回调在 ``on_train_batch_start`` 才写回训练权重），与官方 ``best.pt`` 为 EMA 一致。
"""

from __future__ import annotations

import copy
import math
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn


class ModelEMACallback(pl.Callback):
    """对 ``pl_module.model``（DAGNet）做 EMA；默认 ``decay=0.9999``、``tau=2000``。"""

    def __init__(self, decay: float = 0.9999, tau: int = 2000) -> None:
        super().__init__()
        self.decay = float(decay)
        self.tau = int(tau)
        self.updates = 0
        self.ema: nn.Module | None = None
        self._backup: dict[str, torch.Tensor] | None = None
        self._pending: dict[str, Any] | None = None

    def decay_at(self, updates: int) -> float:
        """``decay * (1 - exp(-updates / tau))``，与 Ultralytics 相同。"""
        return self.decay * (1.0 - math.exp(-float(updates) / float(self.tau)))

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("ModelEMACallback 要求 pl_module.model 为 nn.Module")
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if self._pending:
            self.updates = int(self._pending.get("updates", 0))
            ema_sd = self._pending.get("ema")
            if ema_sd:
                self.ema.load_state_dict(ema_sd)
            self._pending = None

    @torch.no_grad()
    def _update(self, model: nn.Module) -> None:
        assert self.ema is not None
        self.updates += 1
        d = self.decay_at(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.is_floating_point():
                continue
            v.copy_(v * d + (1.0 - d) * msd[k].detach())

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking or self.ema is None:
            return
        acc = int(getattr(trainer, "accumulate_grad_batches", 1) or 1)
        is_last = bool(getattr(trainer, "is_last_batch", False))
        if acc > 1 and (batch_idx + 1) % acc != 0 and not is_last:
            return
        self._update(pl_module.model)

    def _apply_ema(self, pl_module: pl.LightningModule) -> None:
        if self.ema is None or self._backup is not None:
            return
        src = pl_module.model
        self._backup = {k: t.detach().clone() for k, t in src.state_dict().items()}
        src.load_state_dict(self.ema.state_dict())

    def _restore(self, pl_module: pl.LightningModule) -> None:
        if self._backup is None:
            return
        pl_module.model.load_state_dict(self._backup)
        self._backup = None

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._restore(pl_module)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._apply_ema(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply_ema(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore(pl_module)

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._apply_ema(pl_module)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore(pl_module)

    def state_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "ema": None if self.ema is None else self.ema.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.ema is None:
            self._pending = dict(state_dict)
            return
        self.updates = int(state_dict.get("updates", 0))
        ema_sd = state_dict.get("ema")
        if ema_sd:
            self.ema.load_state_dict(ema_sd)
