"""Trainer subclass for LightningCLI custom subcommands."""

from __future__ import annotations

from typing import Optional

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class LovelyTrainer(Trainer):
    """Trainer with extra CLI subcommands (e.g. ``export``).

    **关于 ``fit`` 里的 ``ckpt_path``（LightningCLI / Trainer 默认行为）**

    - 解析阶段：若提供 checkpoint 文件路径，LightningCLI 会尝试用其中的 ``hyper_parameters``
      合并进当前配置（便于恢复与 YAML 不完全一致的 hparams），见
      ``LightningCLI._parse_ckpt_path``。
    - 训练阶段：``Trainer.fit(..., ckpt_path=...)`` 会把 **完整训练状态**（模型权重、优化器、
      epoch、回调等）从该文件恢复，由 ``CheckpointConnector`` 完成。

    **关于本类的 ``export`` 子命令（第一版参数）**

    - 仅 **``export_format``** 与可选 **``ckpt_path``**；ONNX/PT 细节仍来自 YAML ``exporter``。
    - ``ckpt_path`` 经 ``BaseModule.export`` → ``export_dagnet``：只把 checkpoint 里 **``model.`` 前缀**
      的权重加载到 DAGNet，再调用 ``DAGNet.export``；**不**恢复优化器 / epoch（与 ``fit`` 不同）。
    """

    def export(
        self,
        model: LightningModule,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
        export_format: str = "onnx",
    ) -> str:
        """导出模型，返回写出文件路径。"""
        _ = datamodule
        export_fn = getattr(model, "export", None)
        if not callable(export_fn):
            raise TypeError(
                f"{type(model).__name__} 未实现 export(ckpt_path=..., export_format=...)。"
                "请继承 BaseModule 或自行实现。"
            )
        out: str = export_fn(ckpt_path=ckpt_path, export_format=export_format)
        rank_zero_info(f"导出完成: {out}")
        return out

    def prune(
        self,
        model: LightningModule,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> str:
        """剪枝导出，返回写出 ``.pth`` 的绝对路径。

        剪枝超参、输出路径（可选）均在 YAML ``pruner.init_args`` 中配置；
        未设 ``output_path`` 时写出 ``pruning{{率}}_{{stem}}.pth``（与 ``--ckpt_path`` 同目录）。
        """
        _ = datamodule
        prune_fn = getattr(model, "prune", None)
        if not callable(prune_fn):
            raise TypeError(
                f"{type(model).__name__} 未实现 prune(ckpt_path=..., ...)。"
                "请继承 BaseModule 或自行实现。"
            )
        out: str = prune_fn(ckpt_path=ckpt_path)
        rank_zero_info(f"剪枝完成: {out}")
        return out
