from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
import torch.nn as nn

from lovely_deep_learning.export.base import BaseExporter
from lovely_deep_learning.model.weight_loader import DAGNetWeightLoader
from lovely_deep_learning.utils.factory import dynamic_class_instantiate_from_string

if TYPE_CHECKING:
    from lovely_deep_learning.pruning.base import BasePruner


class DAGNet(nn.Module):
    """按有向无环图组装的 ``nn.Module``：根据 ``structure`` 递归实例化各层，``forward`` 按拓扑顺序求值。

    ``forward`` 接收与 ``inputs`` 顺序一致的张量列表，返回 ``outputs`` 中每项 ``from`` 所指节点的输出元组。
    权重可通过构造参数 ``pretrained`` + ``weight`` 加载，或事后调用 :meth:`load_weights`（委托
    :class:`~lovely_deep_learning.model.weight_loader.DAGNetWeightLoader`）。

    ``weight`` 配置::

        map_location: cpu
        strict: false
        stages:
          - format: official
            path: pretrained_models/foo.pth
            opts:
              url: https://...
              src_key_prefix: layers.
              src_key_slice_start: 0

    顶层仅 ``map_location``、``strict``、``stages``；各 ``format`` 专有参数放在对应 ``stages[].opts``。
    """

    def __init__(
        self,
        structure,
        weight=None,
        pretrained=False,
        model_name="undefined",
        exporter: BaseExporter | None = None,
        pruner: "BasePruner | None" = None,
        weight_loader: DAGNetWeightLoader | None = None,
    ):
        super().__init__()

        self._check_config_is_valid(structure)
        self.structure_config = structure
        self.pretrained = pretrained
        self.model_name = model_name
        self.exporter = exporter
        self.pruner = pruner
        self._weight_loader = weight_loader or DAGNetWeightLoader()

        self.inputs = self.structure_config["inputs"]
        self.outputs = self.structure_config["outputs"]
        self.layers_config = self.structure_config["layers"]
        self.layers = nn.ModuleDict()
        self._init_layers()
        if pretrained:
            if not weight:
                raise ValueError(
                    "pretrained=True requires `weight` with non-empty `stages`."
                )
            self.load_weights(**weight)

    def load_weights(self, **cfg: Any) -> None:
        """按 ``weight`` 配置加载（``stages`` 列表，可多步）。"""
        self._weight_loader.load(self, **cfg)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]):
        if isinstance(x, torch.Tensor):
            x = [x]
        outputs = {}
        for i, node in enumerate(self.inputs):
            outputs[node["name"]] = x[i]

        for cfg in self.layers_config:
            name = cfg["name"]
            from_layers = cfg.get("from", [])
            layer = self.layers[name]

            if len(from_layers) == 1:
                inp = outputs[from_layers[0]]
            else:
                inp = [outputs[f] for f in from_layers]

            outputs[name] = layer(inp)

        return tuple(outputs[n["from"][0]] for n in self.outputs)

    @staticmethod
    def _validate_named_nodes(label: str, nodes: Any) -> None:
        if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
            raise TypeError(f"'{label}' must be a list of dicts with at least 'name' key")
        for n in nodes:
            if "name" not in n:
                raise ValueError(f"Each {label} node must have a 'name' key")

    def _check_config_is_valid(self, config):
        if "inputs" not in config:
            raise ValueError("Config must contain 'inputs'")
        self._validate_named_nodes("inputs", config["inputs"])

        if "outputs" not in config:
            raise ValueError("Config must contain 'outputs'")
        self._validate_named_nodes("outputs", config["outputs"])
        for n in config["outputs"]:
            from_layers = n.get("from")
            if not from_layers or len(from_layers) != 1:
                raise ValueError("Each output node must have exactly one 'from' entry")

        if (
            "layers" not in config
            or not isinstance(config["layers"], list)
            or len(config["layers"]) == 0
        ):
            raise ValueError("'layers' must be a non-empty list")
        for layer_cfg in config["layers"]:
            if "name" not in layer_cfg or "module" not in layer_cfg or "from" not in layer_cfg:
                raise ValueError(
                    "Each layer must have 'name', 'module', and 'from' keys"
                )

    def _init_layers(self):
        def build_module(cfg):
            if "children" in cfg:
                children = []
                for child_cfg in cfg["children"]:
                    child_name = child_cfg["name"]
                    children.append((child_name, build_module(child_cfg)))
                return nn.Sequential(OrderedDict(children))
            module = cfg["module"]
            args = cfg.get("args", {})
            return dynamic_class_instantiate_from_string(module, **args)

        for cfg in self.layers_config:
            name = cfg["name"]
            self.layers[name] = build_module(cfg)
