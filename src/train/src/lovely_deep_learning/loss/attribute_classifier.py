from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class AttributeClassifierCriterion(nn.Module):
    """多属性二元交叉熵。``preds[0]`` 与 ``net_out["attr"]`` 都是 ``(N, K)`` logits 和 0/1 标签。

    ``pos_weight`` 长度须等于属性数。省略时每个属性权重相同。
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | Sequence[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(list(pos_weight), dtype=torch.float32)
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(
        self,
        preds: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        *,
        net_out: dict[str, Any],
        net_in: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        _ = net_in
        logits = preds if isinstance(preds, torch.Tensor) else preds[0]
        target = net_out["attr"].to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError(f"属性标签形状 {tuple(target.shape)} 与 logits {tuple(logits.shape)} 不一致")
        pos_weight = self._bce.pos_weight
        if pos_weight is not None and pos_weight.numel() != logits.shape[1]:
            raise ValueError(
                f"pos_weight 长度 {pos_weight.numel()} 与属性数 {logits.shape[1]} 不一致"
            )
        return self._bce(logits, target)
