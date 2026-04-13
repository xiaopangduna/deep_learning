import os
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO

from lovely_deep_learning.utils.factory import dynamic_class_instantiate_from_string


class DAGNet(nn.Module):
    """按有向无环图组装的 ``nn.Module``：根据 ``structure`` 递归实例化各层，``forward`` 按拓扑顺序求值。

    ``forward`` 接收与 ``inputs`` 顺序一致的张量列表，返回 ``outputs`` 中每项 ``from`` 所指节点的输出元组。
    权重可通过构造参数 ``pretrained`` + ``weight`` 加载，或事后调用 :meth:`load_weights`。
    """

    def __init__(
        self,
        structure,
        weight=None,
        pretrained=False,
        model_name="undefined",
        exporter: Any = None,
    ):
        """
        Parameters
        ----------
        structure : dict
            图结构配置，须含：

            - ``inputs``：``list[dict]``，每项至少含 ``name``；可选 ``shape`` 供文档/校验。
            - ``outputs``：``list[dict]``，每项含 ``name`` 与 ``from``（长度为 1 的层名列表，表示该输出取自哪一层）。
            - ``layers``：``list[dict]``，每项含 ``name``、``module``（类或可导入字符串）、``from``（上游层名列表）；
              可选 ``args`` 传入模块构造参数；含 ``children`` 时按 ``nn.Sequential`` 递归构建子模块。

        weight : dict, optional
            仅在 ``pretrained=True`` 时使用；键与 :meth:`load_weights` 一致，例如
            ``path``、``url``、``map_location``、``strict``、``src_key_prefix``、``src_key_slice_start``。

        pretrained : bool
            为 ``True`` 时在构建完 ``layers`` 后调用 ``self.load_weights(**weight)``；此时 ``weight`` 不得为 ``None``。

        model_name : str
            标识名，默认 ``"undefined"``。
        """
        super().__init__()

        self._check_config_is_valid(structure)
        self.structure_config = structure
        self.weight_config = weight
        self.pretrained = pretrained
        self.model_name = model_name
        if isinstance(exporter, dict):
            from lovely_deep_learning.export.yolov8 import YOLOv8Exporter

            self.exporter = YOLOv8Exporter(**exporter)
        else:
            self.exporter = exporter

        self.inputs = self.structure_config["inputs"]
        self.outputs = self.structure_config["outputs"]
        self.layers_config = self.structure_config["layers"]
        self.layers = nn.ModuleDict()
        self._init_layers()
        if pretrained:
            if weight is None:
                raise ValueError("pretrained=True requires a non-None weight config dict.")
            self.load_weights(**self.weight_config)

    def load_weights(
        self,
        path: str,
        url: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = False,
        src_key_prefix: str = "layers.",
        src_key_slice_start: int = 0,
    ) -> None:
        """从本地或 URL 加载权重，按键名截断与前缀规则映射到 ``self.layers.*`` 并 ``load_state_dict``。

        Args:
            path: 权重文件本地路径；不存在且提供 ``url`` 时会下载到此路径。
            url: 可选下载地址。
            map_location: ``torch.load`` 的 ``map_location``。
            strict: 是否严格匹配全部参数。
            src_key_prefix: 映射到 DAGNet 参数名时添加的前缀（默认 ``layers.``）。
            src_key_slice_start: 源键名截断起始下标（如 YOLO 常为 12）。

        Raises:
            ValueError: ``path`` 为空或 ``src_key_slice_start < 0``。
            FileNotFoundError: 本地无文件且未提供 ``url``。
        """
        print(f"============================================================================")
        print(f"DAGNet: Loading weights from {path} (map_location={map_location})")
        print(f"DAGNet: Strict mode: {strict}")
        print(f"DAGNet: Source key prefix: {src_key_prefix}")
        print(f"DAGNet: Source key slice start: {src_key_slice_start}")

        if not path:
            raise ValueError("'path' must be provided as the target file location.")

        if src_key_slice_start < 0:
            raise ValueError("'src_key_slice_start' must be a non-negative integer.")

        final_path = path

        if os.path.isfile(path):
            print(f"DAGNet: Found local weight file at {path}. Using it.")
        elif url:
            print(f"DAGNet: Local file {path} not found. Downloading from {url}...")
            try:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                urllib.request.urlretrieve(url, path)
                print(f"DAGNet: Downloaded weights to {path}.")
            except Exception as e:
                error_msg = f"DAGNet: Failed to download weights from {url} to {path}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            error_msg = (
                f"DAGNet: Weight file not found at {path} and no 'url' provided for download."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            print(f"DAGNet: Loading weights from {final_path} (map_location={map_location})")
            if "yolo" in path.lower():
                source_state_dict: Dict[str, Any] = YOLO(path).state_dict()
                print(f"DAGNet: Loaded weights for YOLO model")
            else:
                source_state_dict = torch.load(final_path, map_location=map_location)
                print(f"DAGNet: Loaded weights for PyTorch model")
            mapped_state_dict: Dict[str, Any] = {}
            model_state_dict = self.state_dict()
            model_state_keys = set(model_state_dict.keys())

            print(
                f"DAGNet: Mapping keys with prefix '{src_key_prefix}' "
                f"and truncating from index {src_key_slice_start}..."
            )
            for src_key, value in source_state_dict.items():
                truncated_key = src_key[src_key_slice_start:]
                dag_net_key = src_key_prefix + truncated_key
                if dag_net_key in model_state_keys:
                    if model_state_dict[dag_net_key].shape == value.shape:
                        mapped_state_dict[dag_net_key] = value
                    else:
                        print(
                            f"Warning: Shape mismatch for key '{dag_net_key}'. "
                            f"Expected: {model_state_dict[dag_net_key].shape}, Got: {value.shape}. Skipping."
                        )
                else:
                    print(
                        f"Info: Mapped key '{dag_net_key}' (from '{src_key}') not found in model. Skipping."
                    )

            print(f"DAGNet: Mapped {len(mapped_state_dict)} compatible keys.")

            print(
                f"DAGNet: Loading {len(mapped_state_dict)} mapped keys into model (strict={strict})..."
            )
            missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=strict)

            if strict and (missing_keys or unexpected_keys):
                print(f"Warning (Strict Mode): Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
            elif not strict:
                if missing_keys:
                    print(f"Info (Non-Strict): Missing keys (in model but not in mapped weights): {missing_keys}")

            print("DAGNet: Weights loading process completed.")
            print(f"============================================================================")

        except Exception as e:
            error_msg = f"DAGNet: Error during loading from {final_path}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def forward(self, x: List[torch.Tensor]):
        outputs = {}
        for i, node in enumerate(self.inputs):
            inp = x[i]
            outputs[node["name"]] = inp

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

    def export(
        self,
        format: str = "onnx",
        exporter: Any = None,
    ) -> str:
        """导出当前 DAGNet 模型。

        Args:
            format: 导出格式，当前仅支持 ``onnx``。
            exporter: 可选导出器实例，优先级高于构造时传入的 ``self.exporter``。
        """
        active_exporter = exporter if exporter is not None else self.exporter
        if active_exporter is None:
            from lovely_deep_learning.export.yolov8 import YOLOv8Exporter

            active_exporter = YOLOv8Exporter()
        if hasattr(active_exporter, "bind"):
            active_exporter.bind(self)
        return active_exporter.export(format=format)

    def _check_config_is_valid(self, config):
        if "inputs" not in config:
            raise ValueError("Config must contain 'inputs'")
        if not isinstance(config["inputs"], list) or not all(
            isinstance(n, dict) for n in config["inputs"]
        ):
            raise TypeError("'inputs' must be a list of dicts with at least 'name' key")
        for n in config["inputs"]:
            if "name" not in n:
                raise ValueError("Each input node must have a 'name' key")

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
