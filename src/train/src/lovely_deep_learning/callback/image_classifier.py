"""图像分类相关 Lightning 回调（TensorBoard 图像记录；测试与 predict 阶段 Writer 落盘与 CSV）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torchvision.utils import make_grid

from ..dataset.image_classifier import ImageClassifierDataset


def _tensorboard_experiment(trainer: pl.Trainer):
    seen: set[int] = set()
    candidates: list[Any] = []
    loggers = getattr(trainer, "loggers", None)
    if loggers:
        candidates.extend(loggers)
    if trainer.logger is not None:
        candidates.insert(0, trainer.logger)
    for lg in candidates:
        oid = id(lg)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(lg, TensorBoardLogger):
            return lg.experiment
    return None


class ImageClassifierTensorBoardImageLogCallback(pl.Callback):
    """将 ``ImageClassifierModule`` 在 ``batch_idx==0`` 返回的 ``tb_sample`` 写入 TensorBoard（记录图像）。"""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_log_sample(trainer, pl_module, outputs, split="train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_log_sample(trainer, pl_module, outputs, split="val")

    def _maybe_log_sample(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any] | None,
        split: str,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        sample = outputs.get("tb_sample")
        if sample is None or sample.get("split") != split:
            return
        experiment = _tensorboard_experiment(trainer)
        if experiment is None:
            return
        try:
            dm = trainer.datamodule
            if split == "train":
                dataset: ImageClassifierDataset = dm.train_dataset
            else:
                dataset = dm.val_dataset
            self._log_images_with_target_and_predictions(
                experiment=experiment,
                global_step=int(pl_module.global_step),
                img=sample["img"],
                net_out=sample["net_out"],
                preds=sample["pred_ids"],
                class_id_conf=sample["pred_conf"],
                dataset=dataset,
                log_prefix=split,
            )
        except Exception as e:
            print(
                f"Warning: TensorBoard image log ({split}) failed at "
                f"global_step={pl_module.global_step}, {e}"
            )

    @staticmethod
    def _log_images_with_target_and_predictions(
        experiment: Any,
        global_step: int,
        img: torch.Tensor,
        net_out: dict[str, Any],
        preds: torch.Tensor,
        class_id_conf: torch.Tensor,
        dataset: ImageClassifierDataset,
        log_prefix: str,
    ) -> None:
        imgs_with_label = []
        for i in range(min(9, img.shape[0])):
            img_tensor = img[i]
            class_name = net_out["class_name"][i]
            class_id = net_out["class_id"][i]
            class_id_pred = preds[i].item()
            class_name_pred = dataset.map_class_id_to_class_name[class_id_pred]
            confidence_pred = class_id_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(img_tensor)

            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=class_name,
                class_id=class_id,
                class_name_pred=class_name_pred,
                class_id_pred=class_id_pred,
                class_id_conf=confidence_pred,
            )
            img_with_label_tensor = dataset.convert_img_from_numpy_to_tensor_uint8(
                img_np
            )
            imgs_with_label.append(img_with_label_tensor)
        img_grid = make_grid(imgs_with_label, nrow=3)
        experiment.add_image(
            f"{log_prefix}/sample_batch", img_grid, global_step=global_step
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
        if self.save_dir is None:
            return
        dataset: ImageClassifierDataset = trainer.test_dataloaders.dataset

        img_tv = batch[0]["img_tv_transformed"]  # [B,3,H,W]

        class_id_pred = outputs["class_id_pred"]
        class_id_conf = outputs["class_id_conf"]

        img_tv = img_tv.detach().cpu()

        B = img_tv.shape[0]

        for i in range(B):
            img_path = Path(batch[0]["img_path"][i])
            img = img_tv[i]  # [3,H,W]

            cur_class_id = batch[1]["class_id"][i].item()
            cur_class_name = batch[1]["class_name"][i]

            cur_class_id_pred = class_id_pred[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = class_id_conf[i].item()

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
        if self.save_dir is None:
            return
        dataset: ImageClassifierDataset = trainer.predict_dataloaders.dataset

        img_tv = batch[0]["img_tv_transformed"]  # [B,3,H,W]

        class_id_pred = outputs["class_id_pred"]
        class_id_conf = outputs["class_id_conf"]

        img_tv = img_tv.detach().cpu()

        B = img_tv.shape[0]

        for i in range(B):
            img_path = Path(batch[0]["img_path"][i])
            img = img_tv[i]  # [3,H,W]
            cur_class_id_pred = class_id_pred[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = class_id_conf[i].item()

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
