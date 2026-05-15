import json
import os
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO

from lovely_deep_learning.export.base import BaseExporter
from lovely_deep_learning.utils.factory import dynamic_class_instantiate_from_string

if TYPE_CHECKING:
    from lovely_deep_learning.pruning.base import BasePruner


@dataclass
class LoadInfo:
    source: str
    load_pruned: bool
    format: str


class DAGNet(nn.Module):
    """按有向无环图组装的 ``nn.Module``：根据 ``structure`` 递归实例化各层，``forward`` 按拓扑顺序求值。

    ``forward`` 接收与 ``inputs`` 顺序一致的张量列表，返回 ``outputs`` 中每项 ``from`` 所指节点的输出元组。
    权重可通过构造参数 ``pretrained`` + ``weight`` 加载，或事后调用 :meth:`load_weights`。
    剪枝与导出分别由 ``pruner``、``exporter`` 策略对象完成（YAML ``class_path`` 注入）。
    """

    def __init__(
        self,
        structure,
        weight=None,
        pretrained=False,
        model_name="undefined",
        exporter: BaseExporter | None = None,
        pruner: "BasePruner | None" = None,
    ):
        super().__init__()

        self._check_config_is_valid(structure)
        self.structure_config = structure
        self.weight_config = weight
        self.pretrained = pretrained
        self.model_name = model_name
        self.exporter = exporter
        self.pruner = pruner

        self.inputs = self.structure_config["inputs"]
        self.outputs = self.structure_config["outputs"]
        self.layers_config = self.structure_config["layers"]
        self.layers = nn.ModuleDict()
        self._init_layers()
        if pretrained:
            self.load_weights(**self.weight_config)

    # ------------------------------------------------------------------ weights
    def load_weights(
        self,
        path: Optional[str] = None,
        path_custom: Optional[str] = None,
        url: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = False,
        src_key_prefix: str = "layers.",
        src_key_slice_start: int = 0,
        load_pruned: bool = False,
    ) -> None:
        """按来源路由加载权重：自定义 / 官方预训练。"""
        if path_custom:
            self.load_from_checkpoint(
                path_custom,
                map_location=map_location,
                strict=strict,
                load_pruned=load_pruned,
            )
            return
        if not path:
            raise ValueError("Either 'path_custom' or 'path' must be provided.")
        self.load_official_weights(
            path=path,
            url=url,
            map_location=map_location,
            strict=strict,
            src_key_prefix=src_key_prefix,
            src_key_slice_start=src_key_slice_start,
        )

    def load_from_checkpoint(
        self,
        source: Union[str, Path],
        *,
        load_pruned: Optional[bool] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        src_key_prefix: str = "layers.",
        src_key_slice_start: int = 0,
    ) -> LoadInfo:
        """统一加载：Lightning ckpt、普通 state_dict、tp 剪枝产物。"""
        print(
            "============================================================================"
        )
        print(f"DAGNet: Loading weights from {source} (map_location={map_location})")
        print(f"DAGNet: strict={strict}, load_pruned={load_pruned}")
        source_path = Path(source) if not isinstance(source, Path) else source
        if not source_path.is_file():
            raise FileNotFoundError(f"DAGNet: Weight file not found: {source_path}")
        try:
            info = self._load_weights_from_file(
                source_path,
                load_pruned=load_pruned,
                map_location=map_location,
                strict=strict,
                src_key_prefix=src_key_prefix,
                src_key_slice_start=src_key_slice_start,
            )
            print("DAGNet: Weights loading process completed.")
            print(
                "============================================================================"
            )
            return info
        except Exception as e:
            msg = f"DAGNet: Error during loading from {source_path}: {e}"
            print(msg)
            raise RuntimeError(msg) from e

    def load_custom_weights(
        self,
        path_custom: str,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = False,
        load_pruned: bool = False,
    ) -> LoadInfo:
        return self.load_from_checkpoint(
            path_custom,
            map_location=map_location,
            strict=strict,
            load_pruned=load_pruned if load_pruned else None,
        )

    @staticmethod
    def read_prune_meta(path: Union[str, Path]) -> Optional[dict[str, Any]]:
        p = Path(path)
        meta_path = (
            p.with_suffix(".meta.json")
            if p.suffix in (".pth", ".pt", ".ckpt")
            else p
        )
        if not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _load_weights_from_file(
        self,
        source_path: Path,
        *,
        load_pruned: Optional[bool],
        map_location: Union[str, torch.device],
        strict: bool,
        src_key_prefix: str,
        src_key_slice_start: int,
    ) -> LoadInfo:
        payload = torch.load(str(source_path), map_location=map_location, weights_only=False)
        is_ckpt = self._is_pl_checkpoint(payload)
        use_tp = self._resolve_load_pruned(
            load_pruned, source_path, payload if is_ckpt else None
        )

        if use_tp and not is_ckpt:
            print(f"DAGNet: tp.load_state_dict <- {source_path}")
            self._tp_load_state_dict(payload)
            return LoadInfo(str(source_path), True, "torch_pruning")

        if is_ckpt:
            weight_cfg = self._weight_cfg_from_ckpt(payload)
            init_path = weight_cfg.get("path_custom")
            if use_tp or weight_cfg.get("load_pruned"):
                if init_path and Path(init_path).is_file():
                    print(f"DAGNet: tp 初始化剪枝结构 <- {init_path}")
                    init_payload = torch.load(
                        str(init_path), map_location=map_location, weights_only=False
                    )
                    self._tp_load_state_dict(init_payload)
                elif weight_cfg.get("load_pruned"):
                    raise FileNotFoundError(
                        "剪枝 ckpt 需在 weight.path_custom 提供 tp.state_dict 产物。"
                    )
            state = self._model_state_dict(payload)
            self.load_state_dict(state, strict=strict)
            return LoadInfo(
                str(source_path),
                bool(use_tp or weight_cfg.get("load_pruned")),
                "lightning_ckpt",
            )

        state = self._model_state_dict(payload)
        if use_tp:
            self._tp_load_state_dict(state)
            return LoadInfo(str(source_path), True, "torch_pruning")

        model_sd = self.state_dict()
        mapped = {
            (src_key_prefix + src_key[src_key_slice_start:]): v
            for src_key, v in state.items()
            if (src_key_prefix + src_key[src_key_slice_start:]) in model_sd
            and model_sd[src_key_prefix + src_key[src_key_slice_start:]].shape == v.shape
        }
        self.load_state_dict(mapped, strict=strict)
        return LoadInfo(str(source_path), False, "state_dict")

    def _resolve_load_pruned(
        self,
        explicit: Optional[bool],
        source_path: Path,
        checkpoint: Optional[dict[str, Any]],
    ) -> bool:
        if explicit is True:
            return True
        meta = self.read_prune_meta(source_path)
        if meta and meta.get("load_pruned"):
            return True
        if explicit is False:
            return False
        if checkpoint and self._weight_cfg_from_ckpt(checkpoint).get("load_pruned"):
            return True
        return False

    @staticmethod
    def _is_pl_checkpoint(payload: Any) -> bool:
        return isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict)

    @staticmethod
    def _weight_cfg_from_ckpt(checkpoint: dict[str, Any]) -> dict[str, Any]:
        model_cfg = (checkpoint.get("hyper_parameters") or {}).get("model") or {}
        weight = model_cfg.get("weight") if isinstance(model_cfg, dict) else None
        return weight if isinstance(weight, dict) else {}

    def _model_state_dict(self, payload: Any) -> Dict[str, Any]:
        if self._is_pl_checkpoint(payload):
            inner = payload["state_dict"]
            prefixed = {
                k.removeprefix("model."): v
                for k, v in inner.items()
                if k.startswith("model.")
            }
            return prefixed or inner
        if isinstance(payload, dict):
            return payload
        raise ValueError("权重格式无效：需要 state_dict 或 Lightning checkpoint。")

    def _tp_load_state_dict(self, payload: Any) -> None:
        import torch_pruning as tp

        state_dict = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
        tp.load_state_dict(self, state_dict=state_dict)

    def load_official_weights(
        self,
        path: str,
        url: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = False,
        src_key_prefix: str = "layers.",
        src_key_slice_start: int = 0,
    ) -> None:
        print(
            "============================================================================"
        )
        print(
            f"DAGNet: Loading official weights from {path} (map_location={map_location})"
        )
        print(f"DAGNet: Strict mode: {strict}")
        if src_key_slice_start < 0:
            raise ValueError("'src_key_slice_start' must be a non-negative integer.")

        if os.path.isfile(path):
            print(f"DAGNet: Found local weight file at {path}. Using it.")
        elif url:
            print(f"DAGNet: Local file {path} not found. Downloading from {url}...")
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            urllib.request.urlretrieve(url, path)
            print(f"DAGNet: Downloaded weights to {path}.")
        else:
            raise FileNotFoundError(
                f"DAGNet: Weight file not found at {path} and no 'url' provided."
            )

        try:
            if "yolo" in path.lower():
                source_state_dict: Dict[str, Any] = YOLO(path).state_dict()
            else:
                payload = torch.load(path, map_location=map_location)
                source_state_dict = self._extract_state_dict(payload)
            mapped_state_dict: Dict[str, Any] = {}
            model_state_dict = self.state_dict()
            for src_key, value in source_state_dict.items():
                truncated_key = src_key[src_key_slice_start:]
                dag_net_key = src_key_prefix + truncated_key
                if (
                    dag_net_key in model_state_dict
                    and model_state_dict[dag_net_key].shape == value.shape
                ):
                    mapped_state_dict[dag_net_key] = value
            self.load_state_dict(mapped_state_dict, strict=strict)
            print("DAGNet: Official weights loading process completed.")
            print(
                "============================================================================"
            )
        except Exception as e:
            raise RuntimeError(f"DAGNet: Error during official loading from {path}: {e}") from e

    @staticmethod
    def _extract_state_dict(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(
            payload["state_dict"], dict
        ):
            return payload["state_dict"]
        if isinstance(payload, dict):
            return payload
        raise ValueError(
            "Weight format invalid: expected a state_dict dict or a dict containing key `state_dict`."
        )

    # ------------------------------------------------------------------ prune / export
    def prune(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """剪枝并导出 tp.state_dict 产物（需配置 ``pruner``）。"""
        if self.pruner is None:
            raise ValueError("pruner is None，请在 YAML 中配置 model.init_args.model.pruner。")
        return self.pruner.prune(
            self, ckpt_path=ckpt_path, output_path=output_path, **kwargs
        )

    def export(self, export_format: str = "onnx") -> str:
        if self.exporter is None:
            raise ValueError("exporter is None, please set exporter in the constructor")
        return self.exporter.export(model=self, export_format=export_format)

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
            raise TypeError("'outputs' must be a list of dicts with at least 'name' key")
        for n in config["outputs"]:
            if "name" not in n:
                raise ValueError("Each output node must have a 'name' key")

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
