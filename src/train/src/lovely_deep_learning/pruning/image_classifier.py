"""图像分类 DAGNet 剪枝策略。"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from lovely_deep_learning.model.DAGNet import DAGNet
from lovely_deep_learning.pruning.base import BasePruner


class ImageClassifierPruner(BasePruner):
    """默认 ignore 分类头；可选按 ``num_classes`` 忽略最后一层 Linear。"""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def collect_ignored_layers(self, dagnet: DAGNet) -> List[nn.Module]:
        ignored: List[nn.Module] = []
        names = list(self.ignored_layer_names)
        if not names:
            for candidate in ("classifier", "fc", "head"):
                if candidate in dagnet.layers:
                    names = [candidate]
                    break
        for name in names:
            if name not in dagnet.layers:
                raise KeyError(f"ignored_layer_names 中的层不存在: {name!r}")
            ignored.append(dagnet.layers[name])

        if self.num_classes is not None:
            for module in dagnet.modules():
                if isinstance(module, nn.Linear) and module.out_features == self.num_classes:
                    if module not in ignored:
                        ignored.append(module)
        return ignored
