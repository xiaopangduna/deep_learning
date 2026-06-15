"""图像分类相关 Lightning 回调（TensorBoard 图像记录；测试与 predict 阶段 Writer 落盘与 CSV）。"""

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


class LogImageClassifierVisualizationCallback(pl.Callback):
    """
    训练 / 验证每个 epoch **各记录一次**分类可视化到 TensorBoard（``train/sample_classifications``、
    ``val/sample_classifications``），均使用 **``batch_idx==0``** 的 batch。

    依赖 :class:`~lovely_deep_learning.module.image_classifier.ImageClassifierModule`：
    ``training_step`` 返回 ``{"loss", "metric_preds", "net_out"}``（``loss`` 供优化）；
    ``validation_step`` 返回 ``{"metric_preds", "net_out"}``（损失仅通过 ``self.log`` 记录）。

    默认 ``max_images=4``、``nrow=2``，拼成 **2×2** 正方形网格。
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


class ImageClassifierTestAndPredictionWriterCallback(pl.Callback):
    """
    将 **test** 与 **predict** 阶段的标注图与结果表写入本地目录（``cv2.imwrite`` + CSV）。
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
            self.csv_table_pred = []  # 初始化为空列表，用于存储预测结果

            os.makedirs(self.save_dir_test, exist_ok=True)
            os.makedirs(self.save_dir_pred, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
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
        """
        每个预测批次结束时保存图像和预测结果
        """
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
