"""LightningCLI subclass registering custom Trainer subcommands."""

from __future__ import annotations

import importlib
import inspect
import os
import warnings
from collections import defaultdict
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningCLI

from lovely_deep_learning.cli.trainer import LovelyTrainer

_Mapping = (dict, Namespace)


class LovelyLightningCLI(LightningCLI):
    """CLI with ``fit`` / ``validate`` / ``test`` / ``predict`` / ``export`` / ``prune``."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trainer_class", LovelyTrainer)
        super().__init__(*args, **kwargs)

    @staticmethod
    def subcommands():
        commands = dict(LightningCLI.subcommands())
        commands["export"] = {"model", "datamodule"}
        commands["prune"] = {"model", "datamodule"}
        return commands

    def before_instantiate_classes(self) -> None:
        """同一实验目录下的多个 logger 共用一个版本号。单个 logger 仍交给 Lightning。"""
        config = self.config[self.subcommand] if self.subcommand and self.subcommand in self.config else self.config
        trainer = config["trainer"] if isinstance(config, _Mapping) and "trainer" in config else None
        if isinstance(trainer, _Mapping) and "logger" in trainer:
            align_logger_versions(trainer["logger"])


def align_logger_versions(logger_config: Any) -> None:
    """按 ``save_dir/name`` 对齐多个 logger 的 ``version``。

    ``name`` 没写时用该类构造函数的默认值；显式 ``null`` 则版本目录就在 ``save_dir`` 下。
    全是 ``null`` 时取已有 ``version_<整数>`` 的最大号加 1。目录不同，或显式版本互相冲突时只警告。
    """
    specs = _logger_specs(logger_config)
    if len(specs) < 2:
        return

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        groups[spec["root"]].append(spec)
    if len(groups) > 1:
        lines = "\n".join(f"  {spec['class_path']}: {spec['root']}" for spec in specs)
        warnings.warn(
            f"多个 logger 的实验目录不一致，版本号按各自目录处理：\n{lines}",
            UserWarning,
            stacklevel=2,
        )

    for group in groups.values():
        if len(group) < 2:
            continue
        explicit = {(type(spec["version"]), spec["version"]) for spec in group if spec["version"] is not None}
        if len(explicit) > 1:
            lines = "\n".join(f"  {spec['class_path']}: {spec['version']!r}" for spec in group)
            warnings.warn(
                f"同一实验目录 {group[0]['root']} 下的 logger 写了不同的 version，保持原配置：\n{lines}",
                UserWarning,
                stacklevel=2,
            )
            continue
        version = next(iter(explicit))[1] if explicit else _next_version(group[0]["root"])
        for spec in group:
            if spec["version"] is None:
                spec["init_args"]["version"] = version


def _logger_specs(logger_config: Any) -> list[dict[str, Any]]:
    if not isinstance(logger_config, list):
        return []
    specs = []
    for item in logger_config:
        init_args = item["init_args"] if isinstance(item, _Mapping) and "init_args" in item else None
        if not isinstance(init_args, _Mapping) or "save_dir" not in init_args or not init_args["save_dir"]:
            continue
        if "name" in init_args:
            name = init_args["name"] or ""
        else:
            name = _default_name(str(item["class_path"]) if "class_path" in item else "")
            if name is None:
                continue
        specs.append(
            {
                "class_path": str(item["class_path"]) if "class_path" in item else "",
                "init_args": init_args,
                "root": os.path.normpath(os.path.join(os.fspath(init_args["save_dir"]), name)),
                "version": init_args["version"] if "version" in init_args else None,
            }
        )
    return specs


def _default_name(class_path: str) -> str | None:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        return None
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
        default = inspect.signature(cls.__init__).parameters["name"].default
    except (AttributeError, KeyError, ModuleNotFoundError, TypeError, ValueError):
        return None
    if default is inspect.Parameter.empty:
        return None
    return default or ""


def _next_version(root: str) -> int:
    """与 Lightning 相同：只认目录 ``version_<整数>``，取最大值加 1。"""
    if not os.path.isdir(root):
        return 0
    versions = []
    for entry in os.scandir(root):
        if not entry.is_dir() or not entry.name.startswith("version_"):
            continue
        number = entry.name.split("_")[1].replace("/", "")
        if number.isdigit():
            versions.append(int(number))
    return max(versions) + 1 if versions else 0
