"""剪枝策略基类（参考 Torch-Pruning torchvision_pruning 示例）。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from lovely_deep_learning.model.DAGNet import DAGNet


class BasePruner(ABC):
    """可插拔剪枝策略，由 DAGNet.pruner 持有并调用。"""

    def __init__(
        self,
        pruning_ratio: float = 0.3,
        global_pruning: bool = True,
        round_to: int = 8,
        ignored_layer_names: Optional[list[str]] = None,
        importance_p: int = 2,
        interactive: bool = False,
        iterative_steps: int = 1,
    ) -> None:
        self.pruning_ratio = pruning_ratio
        self.global_pruning = global_pruning
        self.round_to = round_to
        self.ignored_layer_names = list(ignored_layer_names or [])
        self.importance_p = importance_p
        self.interactive = interactive
        self.iterative_steps = iterative_steps

    @abstractmethod
    def collect_ignored_layers(self, dagnet: DAGNet) -> List[nn.Module]:
        """返回不参与剪枝的 ``nn.Module`` 列表。"""

    def example_inputs(self, dagnet: DAGNet) -> torch.Tensor:
        shape = dagnet.inputs[0].get("shape")
        if not shape:
            raise ValueError("DAGNet.inputs[0] 缺少 shape。")
        return torch.randn(1, *shape)

    def post_prune(self, dagnet: DAGNet) -> None:
        """剪枝后修补静态属性（ViT/Swin 等可覆写）。"""

    def build_pruner(self, dagnet: DAGNet, example_inputs: torch.Tensor, ignored_layers: List[nn.Module]):
        import torch_pruning as tp

        for p in dagnet.parameters():
            p.requires_grad_(True)
        importance = tp.importance.GroupMagnitudeImportance(p=self.importance_p)
        return tp.pruner.BasePruner(
            dagnet,
            example_inputs,
            importance=importance,
            pruning_ratio=self.pruning_ratio,
            ignored_layers=ignored_layers,
            global_pruning=self.global_pruning,
            round_to=self.round_to,
        )

    def run_pruning(self, dagnet: DAGNet, pruner: Any, example_inputs: torch.Tensor) -> tuple[int, int, float, float]:
        import torch_pruning as tp

        base_macs, base_nparams = tp.utils.count_ops_and_params(dagnet, example_inputs)
        if self.interactive:
            for _ in range(self.iterative_steps):
                for group in pruner.step(interactive=True):
                    group.prune()
        else:
            for _ in range(self.iterative_steps):
                pruner.step()
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(dagnet, example_inputs)
        return base_nparams, pruned_nparams, base_macs, pruned_macs

    def default_output_path(self, ckpt_path: Union[str, Path]) -> Path:
        ckpt = Path(ckpt_path)
        return ckpt.parent / f"{ckpt.stem}_pruned.pth"

    def write_meta(
        self,
        output_path: Path,
        *,
        source_ckpt: str,
        dagnet: DAGNet,
        layer_names: Sequence[str],
        base_nparams: int,
        pruned_nparams: int,
        base_macs: float,
        pruned_macs: float,
        pruning_ratio: float,
    ) -> Path:
        import torch_pruning as tp

        meta = {
            "format": "torch_pruning_state_dict",
            "load_pruned": True,
            "source_ckpt": source_ckpt,
            "model_name": getattr(dagnet, "model_name", "undefined"),
            "pruning_ratio": pruning_ratio,
            "global_pruning": self.global_pruning,
            "round_to": self.round_to,
            "ignored_layer_names": list(layer_names),
            "base_nparams": base_nparams,
            "pruned_nparams": pruned_nparams,
            "base_macs": base_macs,
            "pruned_macs": pruned_macs,
            "output_path": str(output_path.resolve()),
            "torch_pruning_version": getattr(tp, "__version__", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return meta_path

    def prune(
        self,
        model: DAGNet,
        *,
        ckpt_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        pruning_ratio: Optional[float] = None,
        global_pruning: Optional[bool] = None,
        round_to: Optional[int] = None,
        ignored_layer_names: Optional[list[str]] = None,
        **_: Any,
    ) -> str:
        if ckpt_path is None:
            raise ValueError("prune 需要 --ckpt_path。")

        ratio = self.pruning_ratio if pruning_ratio is None else pruning_ratio
        if global_pruning is not None:
            self.global_pruning = global_pruning
        if round_to is not None:
            self.round_to = round_to
        if ignored_layer_names is not None:
            self.ignored_layer_names = list(ignored_layer_names)

        model.load_from_checkpoint(ckpt_path, load_pruned=False, strict=False)

        example_inputs = self.example_inputs(model)
        ignored_layers = self.collect_ignored_layers(model)
        layer_names = self.ignored_layer_names or [
            n for n in ("classifier", "fc", "head") if n in model.layers
        ][:1]

        old_ratio = self.pruning_ratio
        self.pruning_ratio = ratio
        tp_pruner = self.build_pruner(model, example_inputs, ignored_layers)
        self.pruning_ratio = old_ratio

        base_nparams, pruned_nparams, base_macs, pruned_macs = self.run_pruning(
            model, tp_pruner, example_inputs
        )
        self.post_prune(model)

        out = Path(output_path) if output_path else self.default_output_path(ckpt_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        import torch_pruning as tp

        model.zero_grad(set_to_none=True)
        torch.save(tp.state_dict(model), str(out))

        meta_path = self.write_meta(
            out,
            source_ckpt=str(ckpt_path),
            dagnet=model,
            layer_names=layer_names,
            base_nparams=base_nparams,
            pruned_nparams=pruned_nparams,
            base_macs=base_macs,
            pruned_macs=pruned_macs,
            pruning_ratio=ratio,
        )

        reduction = 1.0 - pruned_nparams / base_nparams if base_nparams else 0.0
        print(
            f"剪枝完成: {out}\n"
            f"  Params: {base_nparams/1e6:.3f}M -> {pruned_nparams/1e6:.3f}M "
            f"({reduction*100:.1f}% 减少)\n"
            f"  MACs:   {base_macs/1e9:.4f}G -> {pruned_macs/1e9:.4f}G\n"
            f"  Meta:   {meta_path}"
        )
        return str(out.resolve())
