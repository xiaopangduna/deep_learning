"""图像分类 DAGNet 剪枝：逻辑见 :class:`~lovely_deep_learning.pruning.base.BasePruner`。"""

from __future__ import annotations

from lovely_deep_learning.pruning.base import BasePruner


class ImageClassifierPruner(BasePruner):
    """与 :class:`BasePruner` 相同实现；YAML 中单独类名便于与检测等任务的 pruner 区分。"""
