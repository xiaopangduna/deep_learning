import torch
import torch.nn as nn
from collections import OrderedDict
from lovely_deep_learning.utils.factory import dynamic_class_instantiate_from_string


class DAGNet(nn.Module):
    def __init__(self, config):
        """
        config: dict
            input_nodes: list of dict {"name": str, "shape": tuple}
            output_nodes: list of dict {"name": str, "shape": tuple}
            layers: list of dict {
                "name": str,
                "module": nn.Module class or str,
                "args": dict,
                "from": list of input layer names
            }
        """
        super().__init__()

        self._check_config_is_valid(config)

        self.config = config
        self.inputs = config["inputs"]
        self.outputs = config["outputs"]
        self.layers_config = config["layers"]
        self.layers = nn.ModuleDict()

        self._init_layers()

    def forward(self, x:list):
        outputs = {}
        # 初始化输入
        for i, node in enumerate(self.inputs):
            inp = x[i]
            outputs[node["name"]] = inp

        # 遍历所有层
        for cfg in self.layers_config:
            name = cfg["name"]
            from_layers = cfg.get("from", [])
            layer = self.layers[name]

            # 获取输入张量
            if len(from_layers) == 1:
                inp = outputs[from_layers[0]]
            else:
                inp = [outputs[f] for f in from_layers]

            outputs[name] = layer(inp)

        # 返回输出
        return tuple(outputs[n["from"][0]] for n in self.outputs)

    def _check_config_is_valid(self, config):
        # 检查 inputs
        if "inputs" not in config:
            raise ValueError("Config must contain 'inputs'")
        if not isinstance(config["inputs"], list) or not all(
            isinstance(n, dict) for n in config["inputs"]
        ):
            raise TypeError("'inputs' must be a list of dicts with at least 'name' key")
        for n in config["inputs"]:
            if "name" not in n:
                raise ValueError("Each input node must have a 'name' key")

        # 检查 outputs
        if "outputs" not in config:
            raise ValueError("Config must contain 'outputs'")
        if not isinstance(config["outputs"], list) or not all(
            isinstance(n, dict) for n in config["outputs"]
        ):
            raise TypeError(
                "'outputs' must be a list of dicts with at least 'name' key"
            )
        for n in config["outputs"]:
            if "name" not in n:
                raise ValueError("Each output node must have a 'name' key")

        # 检查 layers
        if (
            "layers" not in config
            or not isinstance(config["layers"], list)
            or len(config["layers"]) == 0
        ):
            raise ValueError("'layers' must be a non-empty list")
        for l in config["layers"]:
            if "name" not in l or "module" not in l or "from" not in l:
                raise ValueError(
                    "Each layer must have 'name', 'module', and 'from' keys"
                )

    def _init_layers(self):
        def build_module(cfg):
            """递归构建单个模块"""
            if "children" in cfg:
                # 构建子模块 (nn.Sequential)
                children = []
                for child_cfg in cfg["children"]:
                    child_name = child_cfg["name"]
                    children.append((child_name, build_module(child_cfg)))
                return nn.Sequential(OrderedDict(children))
            else:
                # 普通模块
                module = cfg["module"]
                args = cfg.get("args", {})
                return dynamic_class_instantiate_from_string(module, **args)

        for cfg in self.layers_config:
            name = cfg["name"]
            self.layers[name] = build_module(cfg)