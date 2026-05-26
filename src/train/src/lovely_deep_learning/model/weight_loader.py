"""DAGNet 权重加载：解析 ``weight.stages`` 配置并按 format 写入 ``nn.Module``。"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WEIGHT_FORMATS = frozenset({"official", "dense", "torch_pruning"})
WEIGHT_GLOBAL_KEYS = frozenset({"map_location", "strict"})
WEIGHT_OPTS_BY_FORMAT: Dict[str, frozenset[str]] = {
    "official": frozenset({"url", "src_key_prefix", "src_key_slice_start"}),
    "dense": frozenset({"src_key_prefix", "src_key_slice_start"}),
    "torch_pruning": frozenset(),
}

_STEP_KEYS = frozenset({"format", "path", "opts"})
_DISALLOWED_TOP_LEVEL_WEIGHT_KEYS = ("format", "path", "opts")

StepHandler = Callable[[nn.Module, Path, Dict[str, Any]], None]


class DAGNetWeightLoader:
    """无状态权重加载器；``load(module, **weight_cfg)`` 按 ``stages`` 顺序应用各步。"""

    def __init__(self) -> None:
        self._step_handlers: Dict[str, StepHandler] = {
            "official": self._load_step_official,
            "torch_pruning": self._load_step_torch_pruning,
            "dense": self._load_step_dense,
        }

    def load(self, module: nn.Module, **cfg: Any) -> None:
        steps = self.normalize_steps(cfg)
        for i, step in enumerate(steps):
            logger.info(
                "DAGNet: weight step %d/%d format=%s path=%s",
                i + 1,
                len(steps),
                step["format"],
                step["path"],
            )
            self._apply_step(module, step)

    @classmethod
    def normalize_steps(cls, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not cfg:
            raise ValueError("weight config must not be empty.")

        cfg = dict(cfg)
        disallowed = [k for k in _DISALLOWED_TOP_LEVEL_WEIGHT_KEYS if k in cfg]
        if disallowed:
            keys = ", ".join(repr(k) for k in disallowed)
            raise ValueError(
                f"weight: top-level {keys} not allowed; "
                "use weight.stages (each step needs format and path)."
            )

        global_opts = {k: cfg[k] for k in WEIGHT_GLOBAL_KEYS if k in cfg}
        allowed_top = WEIGHT_GLOBAL_KEYS | {"stages"}
        unknown = set(cfg.keys()) - allowed_top
        if unknown:
            raise ValueError(
                f"weight: unknown top-level keys {sorted(unknown)}; "
                f"allowed: {sorted(allowed_top)}."
            )

        raw_stages = cfg.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("weight must include non-empty 'stages'.")
        return [
            {**global_opts, **cls._normalize_step_dict(s, i)}
            for i, s in enumerate(raw_stages)
        ]

    @classmethod
    def _normalize_step_dict(cls, step: Any, index: int) -> Dict[str, Any]:
        if not isinstance(step, dict):
            raise TypeError(f"weight.stages[{index}] must be a dict.")
        fmt = step.get("format")
        path = step.get("path")
        if not fmt or not path:
            raise ValueError(f"weight.stages[{index}] requires 'format' and 'path'.")
        if fmt not in WEIGHT_FORMATS:
            raise ValueError(
                f"weight.stages[{index}].format {fmt!r} invalid; "
                f"expected one of {sorted(WEIGHT_FORMATS)}."
            )

        extra = set(step.keys()) - _STEP_KEYS
        if extra:
            raise ValueError(
                f"weight.stages[{index}]: unexpected keys {sorted(extra)}; "
                "put format-specific options under 'opts'."
            )

        opts = step.get("opts")
        if opts is None:
            opts = {}
        elif not isinstance(opts, dict):
            raise TypeError(f"weight.stages[{index}].opts must be a dict.")

        cls._validate_step_opts(fmt, opts, index)
        return {"format": fmt, "path": path, **opts}

    @classmethod
    def _validate_step_opts(cls, fmt: str, opts: Dict[str, Any], index: int) -> None:
        for key in opts:
            if key in WEIGHT_GLOBAL_KEYS:
                raise ValueError(
                    f"weight.stages[{index}].opts.{key} belongs at weight top level "
                    f"(map_location / strict), not in opts."
                )
            if key not in WEIGHT_OPTS_BY_FORMAT[fmt]:
                raise ValueError(
                    f"weight.stages[{index}].opts.{key} is not valid for format {fmt!r}; "
                    f"allowed: {sorted(WEIGHT_OPTS_BY_FORMAT[fmt])}."
                )

    def _apply_step(self, module: nn.Module, step: Dict[str, Any]) -> None:
        fmt = step["format"]
        path = Path(step["path"])

        if fmt != "official" and not path.is_file():
            raise FileNotFoundError(f"DAGNet: Weight file not found: {path}")

        try:
            handler = self._step_handlers[fmt]
        except KeyError as e:
            raise ValueError(f"Unhandled weight format: {fmt!r}") from e
        handler(module, path, step)

    @staticmethod
    def _step_load_settings(step: Dict[str, Any]) -> tuple[Any, bool, str, int]:
        return (
            step.get("map_location", "cpu"),
            step.get("strict", False),
            step.get("src_key_prefix", "layers."),
            int(step.get("src_key_slice_start", 0)),
        )

    @staticmethod
    def _torch_load(path: Union[str, Path], map_location: Any) -> Any:
        return torch.load(str(path), map_location=map_location, weights_only=False)

    @staticmethod
    def _ensure_official_file(path: str, url: str | None) -> None:
        if os.path.isfile(path):
            logger.info("DAGNet: using local official weights at %s", path)
            return
        if url:
            logger.info("DAGNet: downloading official weights from %s", url)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            urllib.request.urlretrieve(url, path)
            return
        raise FileNotFoundError(
            f"DAGNet: Weight file not found at {path} and no 'url' provided."
        )

    @staticmethod
    def _checkpoint_state_dict(
        payload: Any, *, strip_model_prefix: bool
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("权重格式无效：需要 state_dict 或 Lightning checkpoint。")
        inner = payload.get("state_dict")
        if isinstance(inner, dict):
            if strip_model_prefix:
                prefixed = {
                    k.removeprefix("model."): v
                    for k, v in inner.items()
                    if k.startswith("model.")
                }
                return prefixed or inner
            return inner
        return payload

    @staticmethod
    def _is_pl_checkpoint(payload: Any) -> bool:
        return isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict)

    @staticmethod
    def _map_state_dict(
        module: nn.Module,
        source: Dict[str, Any],
        *,
        prefix: str,
        slice_start: int,
    ) -> Dict[str, Any]:
        model_sd = module.state_dict()
        return {
            (prefix + src_key[slice_start:]): v
            for src_key, v in source.items()
            if (dst_key := prefix + src_key[slice_start:]) in model_sd
            and model_sd[dst_key].shape == v.shape
        }

    def _load_mapped_state_dict(
        self, module: nn.Module, source: Dict[str, Any], step: Dict[str, Any]
    ) -> None:
        _, strict, prefix, slice_start = self._step_load_settings(step)
        if slice_start < 0:
            raise ValueError("'src_key_slice_start' must be a non-negative integer.")
        mapped = self._map_state_dict(
            module, source, prefix=prefix, slice_start=slice_start
        )
        module.load_state_dict(mapped, strict=strict)

    def _load_step_official(
        self, module: nn.Module, _path: Path, step: Dict[str, Any]
    ) -> None:
        file_path = step["path"]
        map_location, _, _, _ = self._step_load_settings(step)
        self._ensure_official_file(file_path, step.get("url"))

        if "yolo" in file_path.lower():
            from ultralytics import YOLO

            source_state_dict = YOLO(file_path).state_dict()
        else:
            payload = self._torch_load(file_path, map_location)
            source_state_dict = self._checkpoint_state_dict(
                payload, strip_model_prefix=False
            )

        self._load_mapped_state_dict(module, source_state_dict, step)

    def _load_step_torch_pruning(
        self, module: nn.Module, source_path: Path, step: Dict[str, Any]
    ) -> None:
        import torch_pruning as tp

        map_location, _, _, _ = self._step_load_settings(step)
        payload = self._torch_load(source_path, map_location)
        state_dict = self._checkpoint_state_dict(payload, strip_model_prefix=False)
        tp.load_state_dict(module, state_dict=state_dict)

    def _load_step_dense(
        self, module: nn.Module, source_path: Path, step: Dict[str, Any]
    ) -> None:
        map_location, strict, _, _ = self._step_load_settings(step)
        payload = self._torch_load(source_path, map_location)
        if self._is_pl_checkpoint(payload):
            state = self._checkpoint_state_dict(payload, strip_model_prefix=True)
            module.load_state_dict(state, strict=strict)
            return
        state = self._checkpoint_state_dict(payload, strip_model_prefix=False)
        self._load_mapped_state_dict(module, state, step)
