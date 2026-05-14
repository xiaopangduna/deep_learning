"""LightningCLI subclass registering custom Trainer subcommands."""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from lovely_deep_learning.cli.trainer import LovelyTrainer


class LovelyLightningCLI(LightningCLI):
    """CLI with ``fit`` / ``validate`` / ``test`` / ``predict`` / ``export``."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("trainer_class", LovelyTrainer)
        super().__init__(*args, **kwargs)

    @staticmethod
    def subcommands():
        commands = dict(LightningCLI.subcommands())
        commands["export"] = {"model", "datamodule"}
        return commands
