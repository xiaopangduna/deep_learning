"""图像分类 Lightning 回调。

- ``Log*TrainVal*``：train / val → TensorBoard 图像
- ``Save*TestPredict*``：test / predict → 本地目录（标注图 + CSV）

两类 Callback 均依赖 :class:`~lovely_deep_learning.module.base.BaseModule` 的 step 返回值
（``metric_preds`` / ``net_out``）及 collate 后的 ``net_in: tuple[dict]``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import lightning.pytorch as pl
import pandas as pd
import torch
from torchvision.utils import make_grid

from ..dataset.image_classifier import ImageClassifierDataset


class LogImageClassifierTrainValVisualizationCallback(pl.Callback):
    """train / val 阶段将分类可视化写入 TensorBoard（不落盘）。

    **触发时机**

    - :meth:`on_train_batch_end` / :meth:`on_validation_batch_end`（val 仅 ``dataloader_idx==0``）
    - 仅 ``trainer.is_global_zero`` 且 ``batch_idx==0`` 时执行（每 epoch train/val 各记一次）
    - TensorBoard tag：``train/sample_classifications``、``val/sample_classifications``

    **调用链**

    ``on_*_batch_end`` → :meth:`_try_log_batch` → :meth:`_log_sample_classifications_tensorboard`

    **数据契约**

    Module（:class:`~lovely_deep_learning.module.image_classifier.ImageClassifierModule`，
    继承 :class:`~lovely_deep_learning.module.base.BaseModule`）：

    - ``training_step`` → ``{"loss", "metric_preds", "net_out"}``
    - ``validation_step`` → ``{"metric_preds", "net_out"}``
    - ``metric_preds["pred_ids"]`` / ``["pred_conf"]``：postprocess 的 batch 级预测
    - ``net_out["class_id"]`` / ``["class_name"]``：collate 后的 GT（batched dict）

    Batch（:class:`ImageClassifierDataset` collate）：``(net_in, net_out)``；
    ``net_in`` 为 ``tuple[dict]``，绘制时取 ``net_in[i]["img_tv_transformed"]``。

    Datamodule：``trainer.datamodule.train_dataset`` / ``val_dataset``（:class:`ImageClassifierDataset`），
    提供 ``map_class_id_to_class_name`` 及 tensor ↔ numpy 绘制工具。

    Logger：``pl_module.logger.experiment`` 须支持 ``add_image``（如 TensorBoardLogger）；
    无 logger 或不支持时静默跳过。

    **参数**

    ``max_images``（默认 4）、``nrow``（默认 2）：经 :func:`~torchvision.utils.make_grid` 拼成网格后写入。
    """

    def __init__(
        self,
        max_images: int = 4,
        nrow: int = 2,
    ) -> None:
        super().__init__()
        self.max_images = int(max_images)
        self.nrow = int(nrow)

    def _log_sample_classifications_tensorboard(
        self,
        pl_module: pl.LightningModule,
        batch: Any,
        outputs: dict[str, Any],
        dataset: ImageClassifierDataset,
        tb_tag: str,
    ) -> None:
        """绘制单 batch 样本面板并 ``experiment.add_image``。

        数据契约见 :class:`LogImageClassifierTrainValVisualizationCallback`。
        ``dataset`` 由 :meth:`_try_log_batch` 从 datamodule 注入，使用：

        - :meth:`~lovely_deep_learning.dataset.image_classifier.ImageClassifierDataset.convert_img_from_tensor_to_numpy`
        - :meth:`~lovely_deep_learning.dataset.image_classifier.ImageClassifierDataset.draw_target_and_predict_label_on_numpy`
        - :meth:`~lovely_deep_learning.dataset.base.BaseDataset.convert_img_from_numpy_to_tensor_uint8`
        """
        metric_preds = outputs.get("metric_preds")
        net_out = outputs.get("net_out")
        if metric_preds is None or net_out is None:
            return
        if getattr(pl_module, "logger", None) is None:
            return
        try:
            experiment = pl_module.logger.experiment
        except Exception:
            return
        if not hasattr(experiment, "add_image"):
            return

        net_in, _ = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        n = min(self.max_images, len(net_in))
        if n == 0:
            return

        panels: list[torch.Tensor] = []
        for i in range(n):
            class_name = net_out["class_name"][i]
            class_id = net_out["class_id"][i]
            class_id_pred = pred_ids[i].item()
            class_name_pred = dataset.map_class_id_to_class_name[class_id_pred]
            confidence_pred = pred_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(
                net_in[i]["img_tv_transformed"]
            )
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=class_name,
                class_id=class_id,
                class_name_pred=class_name_pred,
                class_id_pred=class_id_pred,
                class_id_conf=confidence_pred,
            )
            panels.append(dataset.convert_img_from_numpy_to_tensor_uint8(img_np))

        img_grid = make_grid(panels, nrow=self.nrow)
        experiment.add_image(
            tb_tag,
            img_grid,
            global_step=pl_module.global_step,
        )

    def _try_log_batch(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataset_name: str,
        tb_tag: str,
        phase_label: str,
    ) -> None:
        """rank0 / ``batch_idx==0`` 门禁；通过后调用 :meth:`_log_sample_classifications_tensorboard`。

        数据契约见 :class:`LogImageClassifierTrainValVisualizationCallback`。
        ``dataset_name`` 为 ``"train_dataset"`` 或 ``"val_dataset"``。
        """
        if not trainer.is_global_zero or batch_idx != 0:
            return
        if outputs is None or not isinstance(outputs, dict):
            return
        dm = trainer.datamodule
        if dm is None:
            return
        dataset = getattr(dm, dataset_name, None)
        if dataset is None:
            return
        try:
            self._log_sample_classifications_tensorboard(
                pl_module, batch, outputs, dataset, tb_tag
            )
        except Exception as e:
            print(
                f"Warning: failed to log {phase_label} classification visualization at step "
                f"{pl_module.global_step}, {e}"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._try_log_batch(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            "train_dataset",
            "train/sample_classifications",
            "train",
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0:
            return
        self._try_log_batch(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            "val_dataset",
            "val/sample_classifications",
            "val",
        )


class SaveImageClassifierTestPredictVisualizationCallback(pl.Callback):
    """test / predict 阶段将分类可视化与结果表写入本地（``cv2.imwrite`` + CSV）。

    与 :class:`LogImageClassifierTrainValVisualizationCallback` 分工：后者仅 train/val → TensorBoard。

    **触发时机**

    - :meth:`on_test_batch_end`：逐 batch、逐样本写图与 CSV 行
    - :meth:`on_predict_batch_end`：逐 batch、逐样本写图与 CSV 行（无 GT）
    - :meth:`on_test_end` / :meth:`on_predict_end`：汇总写入 CSV 文件
    - ``save_dir`` 为 ``None`` 时不写入

    **数据契约**

    Module（:class:`~lovely_deep_learning.module.base.BaseModule`）：

    - ``test_step`` → ``{"metric_preds", "net_out"}``
    - ``predict_step`` → ``{"metric_preds"}``
    - ``metric_preds["pred_ids"]`` / ``["pred_conf"]``：batch 级张量

    Batch（:class:`ImageClassifierDataset` collate）：``(net_in, net_out)``；
    ``net_in[i]["img_path"]`` / ``["img_tv_transformed"]`` 用于读图与绘制；
    test 阶段 GT 来自 ``net_out["class_id"]`` / ``["class_name"]``。

    Dataloader dataset：``trainer.test_dataloaders.dataset`` /
    ``trainer.predict_dataloaders.dataset``（:class:`ImageClassifierDataset`），
    供类名映射与绘制工具。

    **目录结构**（``save_dir`` 非空时）::

        {save_dir}/test/*.jpg          # 测试标注图
        {save_dir}/test_results.csv    # 测试汇总
        {save_dir}/predict/*.jpg       # 预测标注图
        {save_dir}/prediction_results.csv

    **参数**

    ``test_only_save_mistake=True`` 时，test 阶段仅保存预测错误样本图（CSV 仍记录全部样本）。
    """

    def __init__(self, save_dir: str = None, test_only_save_mistake=True):
        """
        Args:
            save_dir: 写入根目录；为 ``None`` 时不写入。
            test_only_save_mistake: 测试时是否仅保存预测错误样本图。
        """
        super().__init__()
        self.test_only_save_mistake = test_only_save_mistake
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir_test = self.save_dir / "test"
            self.save_dir_pred = self.save_dir / "predict"
            self.csv_table_test = []
            self.csv_table_pred = []

            os.makedirs(self.save_dir_test, exist_ok=True)
            os.makedirs(self.save_dir_pred, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """逐样本绘制 GT + 预测标签，按 ``test_only_save_mistake`` 决定是否写图。"""
        if self.save_dir is None or outputs is None:
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ImageClassifierDataset = trainer.test_dataloaders.dataset

        net_in, net_out = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        B = len(net_in)

        for i in range(B):
            img_path = Path(net_in[i]["img_path"])
            img = net_in[i]["img_tv_transformed"]

            cur_class_id = net_out["class_id"][i].item()
            cur_class_name = net_out["class_name"][i]

            cur_class_id_pred = pred_ids[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = pred_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=cur_class_name,
                class_id=cur_class_id,
                class_name_pred=cur_class_name_pred,
                class_id_pred=cur_class_id_pred,
                class_id_conf=cur_confidence_pred,
            )

            save_path = self.save_dir_test / (img_path.stem + ".jpg")


            self.csv_table_test.append(
                {
                    "img_path": str(img_path),
                    "class_id": cur_class_id,
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "class_name": cur_class_name,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": str(save_path),
                }
            )
            
            if self.test_only_save_mistake and cur_class_id == cur_class_id_pred:
                continue
            cv2.imwrite(save_path, img_np)

    def on_test_epoch_start(self, trainer, pl_module):
        self.csv_table_test = []
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.csv_table_test:
            df = pd.DataFrame(self.csv_table_test)
            csv_save_path = self.save_dir / "test_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """预测阶段无 GT，仅绘制预测标签并写入 ``predict/``。"""
        if self.save_dir is None or outputs is None:
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ImageClassifierDataset = trainer.predict_dataloaders.dataset

        net_in, _net_out = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        B = len(net_in)

        for i in range(B):
            img_path = Path(net_in[i]["img_path"])
            img = net_in[i]["img_tv_transformed"]
            cur_class_id_pred = pred_ids[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = pred_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np, class_name_pred=cur_class_name_pred, class_id_pred=cur_class_id_pred, class_id_conf=cur_confidence_pred
            )

            save_path = self.save_dir_pred / (img_path.stem + ".jpg")
            cv2.imwrite(save_path, img_np)

            self.csv_table_pred.append(
                {
                    "img_path": str(img_path),
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": str(save_path),
                }
            )

    def on_predict_epoch_start(self, trainer, pl_module):
        self.csv_table_pred = []
        return super().on_predict_epoch_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.csv_table_pred:
            df = pd.DataFrame(self.csv_table_pred)
            csv_save_path = self.save_dir / "prediction_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")
