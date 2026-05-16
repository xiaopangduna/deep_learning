"""剪枝策略基类。

- ``importance_cfg`` / ``tp_pruner_cfg``：``class_name`` + ``init_args``，对应 torch_pruning。
- ``ignored_layer_names``：DAGNet ``layers`` 顶层键，剪枝时转为 ``ignored_layers`` 模块列表。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch_pruning as tp

from lovely_deep_learning.model.DAGNet import DAGNet


def _tp_component(cfg: Any, module: Any, *, field: str) -> tuple[type, dict[str, Any]]:
    if isinstance(cfg, dict):
        d = cfg
    else:
        try:
            from jsonargparse import Namespace
        except ImportError:
            Namespace = ()  # type: ignore[misc, assignment]
        if Namespace and isinstance(cfg, Namespace):
            d = cfg.as_dict()
        else:
            raise TypeError(f"{field} 须为 dict 或 Namespace，实际为 {type(cfg).__name__}")
    name = d["class_name"]
    args = dict(d.get("init_args") or {})
    try:
        return getattr(module, name), args
    except AttributeError as e:
        raise ValueError(f"{field} 未知 class_name: {name!r}") from e


class BasePruner:
    """可插拔剪枝策略，由 DAGNet.pruner 持有并调用。"""

    def __init__(
        self,
        importance_cfg: Any,
        tp_pruner_cfg: Any,
        ignored_layer_names: list[str],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self.output_path = Path(output_path).resolve() if output_path else None

        imp_cls, imp_args = _tp_component(importance_cfg, tp.importance, field="importance_cfg")
        self.importance = imp_cls(**imp_args)

        self.ignored_layer_names = list(ignored_layer_names)
        if not self.ignored_layer_names:
            raise ValueError("ignored_layer_names 不能为空。")

        pruner_cls, args = _tp_component(tp_pruner_cfg, tp.pruner, field="tp_pruner_cfg")
        self._pruner_cls = pruner_cls
        self.tp_pruner_cfg = args

    def collect_ignored_layers(self, dagnet: DAGNet) -> List[nn.Module]:
        ignored: List[nn.Module] = []
        for name in self.ignored_layer_names:
            if name not in dagnet.layers:
                raise KeyError(f"ignored_layer_names 中的层不存在: {name!r}")
            ignored.append(dagnet.layers[name])
        return ignored

    def example_inputs(self, dagnet: DAGNet) -> torch.Tensor:
        shape = dagnet.inputs[0].get("shape")
        if not shape:
            raise ValueError("DAGNet.inputs[0] 缺少 shape。")
        return torch.randn(1, *shape)

    def post_prune(self, dagnet: DAGNet) -> None:
        """剪枝后修补静态属性（ViT/Swin 等可覆写）。"""

    def build_pruner(
        self, dagnet: DAGNet, example_inputs: torch.Tensor, ignored_layers: List[nn.Module]
    ):
        for p in dagnet.parameters():
            p.requires_grad_(True)
        return self._pruner_cls(
            dagnet,
            example_inputs=example_inputs,
            importance=self.importance,
            ignored_layers=ignored_layers,
            **self.tp_pruner_cfg,
        )

    def run_pruning(
        self, dagnet: DAGNet, pruner: Any, example_inputs: torch.Tensor
    ) -> tuple[int, int, float, float]:
        base_macs, base_nparams = tp.utils.count_ops_and_params(dagnet, example_inputs)
        for _ in range(pruner.iterative_steps):
            pruner.step()
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(dagnet, example_inputs)
        return base_nparams, pruned_nparams, base_macs, pruned_macs

    @staticmethod
    def pruning_ratio_filename_tag(ratio: float) -> str:
        return f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")

    @classmethod
    def default_output_path_for_ratio(
        cls, ckpt_path: Union[str, Path], pruning_ratio: float
    ) -> Path:
        ckpt = Path(ckpt_path).resolve()
        tag = cls.pruning_ratio_filename_tag(pruning_ratio)
        return ckpt.parent / f"pruning{tag}_{ckpt.stem}.pth"

    def default_output_path(
        self, ckpt_path: Union[str, Path], *, pruning_ratio: float
    ) -> Path:
        return type(self).default_output_path_for_ratio(ckpt_path, pruning_ratio)

    def prune(self, model: DAGNet, *, ckpt_path: Union[str, Path]) -> str:
        """执行剪枝。``ckpt_path`` 由 CLI ``--ckpt_path`` 传入；其余见 YAML ``init_args``。"""
        ratio = float(self.tp_pruner_cfg["pruning_ratio"])

        model.load_from_checkpoint(ckpt_path, load_pruned=False, strict=False)
        example_inputs = self.example_inputs(model)
        ignored_layers = self.collect_ignored_layers(model)
        tp_pruner = self.build_pruner(model, example_inputs, ignored_layers)

        base_nparams, pruned_nparams, base_macs, pruned_macs = self.run_pruning(
            model, tp_pruner, example_inputs
        )
        self.post_prune(model)

        out = self.output_path or self.default_output_path(ckpt_path, pruning_ratio=ratio)
        out.parent.mkdir(parents=True, exist_ok=True)

        model.zero_grad(set_to_none=True)
        torch.save(tp.state_dict(model), str(out))

        reduction = 1.0 - pruned_nparams / base_nparams if base_nparams else 0.0
        print(
            f"剪枝完成: {out}\n"
            f"  Params: {base_nparams/1e6:.3f}M -> {pruned_nparams/1e6:.3f}M "
            f"({reduction*100:.1f}% 减少)\n"
            f"  MACs:   {base_macs/1e9:.4f}G -> {pruned_macs/1e9:.4f}G"
        )
        return str(out.resolve())
