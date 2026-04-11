import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from torchvision.utils import make_grid

from ..dataset.object_detect import (
    ObjectDetectDataset,
    boxes_layout_to_xyxy_numpy,
    greedy_match_pred_gt_iou,
    metric_pred_item_to_xyxy_cls_score_numpy,
)
from ..utils.io import (
    instances_to_json_str,
    write_row_dicts_to_csv_path_skip_if_empty,
)
from ..utils.plot import hstack_bgr_left_right_resize_right_to_match_left_with_gray_separator


def _stack_visual_imgs_from_net_in(net_in: Any) -> torch.Tensor:
    """tuple batch：按样本选用 ``img_tv_transformed`` 或 ``img``（predict 可能仅有 ``img``）。"""
    if isinstance(net_in, dict):
        key = "img_tv_transformed" if "img_tv_transformed" in net_in else "img"
        return net_in[key]
    first = net_in[0]
    key = "img_tv_transformed" if "img_tv_transformed" in first else "img"
    return torch.stack([ni[key] for ni in net_in], dim=0)


def _cpu_imgs_and_paths_from_net_in(
    net_in: Any,
    *,
    tuple_use_tv_stack: bool,
) -> tuple[torch.Tensor, list[Any]]:
    """
    collate 后的 ``net_in`` → CPU 上 ``(B,C,H,W)`` 与路径列表。

    ``tuple_use_tv_stack=True``：tuple batch 时只堆叠 ``img_tv_transformed``（test）；
    ``False``：与 :func:`_stack_visual_imgs_from_net_in` 一致（predict）。
    """
    if isinstance(net_in, dict):
        img = net_in.get("img_tv_transformed", net_in.get("img"))
        paths = net_in["img_path"]
        return img.detach().cpu(), paths
    if tuple_use_tv_stack:
        stacked = torch.stack(
            [
                ni["img_tv_transformed"].data
                if hasattr(ni["img_tv_transformed"], "data")
                else ni["img_tv_transformed"]
                for ni in net_in
            ],
            dim=0,
        )
    else:
        stacked = _stack_visual_imgs_from_net_in(net_in)
    paths = [ni["img_path"] for ni in net_in]
    return stacked.detach().cpu(), paths


def _net_out_sample(net_out: Any, i: int) -> dict[str, Any]:
    if isinstance(net_out, dict):
        return {k: net_out[k][i] for k in net_out}
    return net_out[i]


def _gt_xyxy_cls_numpy(gt: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """单样本 ``net_out[i]``（collate 后一项）→ GT xyxy numpy、类别 id。"""
    if not gt or "bboxes_xyxy_abs_tv_transformed" not in gt:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    bb = gt["bboxes_xyxy_abs_tv_transformed"]
    cl = gt["cls_tv_transformed"]
    if torch.is_tensor(bb):
        bb = bb.detach().cpu().float().numpy()
    if torch.is_tensor(cl):
        cl = cl.detach().cpu().numpy().astype(np.int64).reshape(-1)
    bb = np.asarray(bb, dtype=np.float32).reshape(-1, 4)
    cl = np.asarray(cl).reshape(-1).astype(np.int64)
    return bb, cl


def _map_pred_box_format(pl_module: pl.LightningModule) -> str:
    post = getattr(pl_module, "postprocess", None)
    if post is not None and hasattr(post, "map_pred_box_format"):
        return str(post.map_pred_box_format)
    return "xyxy"


class SaveObjectDetectVisualizationCallback(pl.Callback):
    """保存目标检测可视化（仅 test / predict；训练与验证不写入）。测试阶段使用 :meth:`ObjectDetectDataset.draw_target_and_predict_label_on_numpy`（左=预测，右=GT，IoU 配色）；预测阶段为左=原图、右=预测框。"""

    def __init__(
        self,
        save_dir: str | None = None,
        test_only_save_mistake: bool = True,
        conf_thres: float = 0.25,
        match_iou_thres: float = 0.5,
    ):
        super().__init__()
        self.test_only_save_mistake = test_only_save_mistake
        self.conf_thres = conf_thres
        self.match_iou_thres = match_iou_thres
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_dir_test = None
        self.save_dir_pred = None
        self.csv_table_test: list = []
        self.csv_table_pred: list = []
        if self.save_dir is not None:
            self.save_dir_test = self.save_dir / "test"
            self.save_dir_pred = self.save_dir / "predict"
            os.makedirs(self.save_dir_test, exist_ok=True)
            os.makedirs(self.save_dir_pred, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.save_dir is None or outputs is None:
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ObjectDetectDataset = trainer.test_dataloaders.dataset
        net_in, net_out = batch
        box_fmt = _map_pred_box_format(pl_module)
        img_tv, img_paths = _cpu_imgs_and_paths_from_net_in(
            net_in, tuple_use_tv_stack=True
        )
        B = len(metric_preds)

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]

            pred_xyxy, pred_cls, pred_score = metric_pred_item_to_xyxy_cls_score_numpy(
                metric_preds[i], box_fmt, self.conf_thres
            )

            boxes, cls_ids = _gt_xyxy_cls_numpy(_net_out_sample(net_out, i))

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            names = dataset.map_class_id_to_class_name or None
            pred_order = (
                np.argsort(-pred_score) if pred_xyxy.shape[0] else None
            )
            pred_ok, gt_ok = greedy_match_pred_gt_iou(
                pred_xyxy.astype(np.float32),
                pred_cls.astype(np.int32),
                boxes.astype(np.float32),
                cls_ids.astype(np.int32),
                float(self.match_iou_thres),
                pred_order=pred_order,
            )
            is_match = bool(np.all(pred_ok) and np.all(gt_ok))
            img_out = ObjectDetectDataset.draw_target_and_predict_label_on_numpy(
                img_np,
                pred_xyxy.astype(np.float32),
                pred_cls.astype(np.int32),
                boxes.astype(np.float32),
                cls_ids.astype(np.int32),
                class_names=names,
                pred_scores=pred_score,
                match_by_iou=True,
                iou_match_threshold=float(self.match_iou_thres),
                cached_match=(pred_ok, gt_ok),
            )

            save_path = self.save_dir_test / (img_path.stem + ".jpg")
            if self.test_only_save_mistake and is_match:
                pass
            else:
                cv2.imwrite(str(save_path), img_out)

            self.csv_table_test.append(
                {
                    "img_path": str(img_path),
                    "ground_truth": instances_to_json_str(
                        boxes.astype(np.float32),
                        cls_ids.astype(np.int32),
                        names,
                        scores=None,
                    ),
                    "predictions": instances_to_json_str(
                        pred_xyxy, pred_cls, names, scores=pred_score
                    ),
                    "num_gt": int(boxes.shape[0]),
                    "num_pred": int(pred_xyxy.shape[0]),
                    "match_ok": bool(is_match),
                    "save_path": str(save_path),
                }
            )

    def on_test_epoch_start(self, trainer, pl_module):
        self.csv_table_test = []
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.save_dir is not None:
            write_row_dicts_to_csv_path_skip_if_empty(
                self.save_dir / "test_results.csv", self.csv_table_test
            )

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.save_dir is None or outputs is None:
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ObjectDetectDataset = trainer.predict_dataloaders.dataset
        net_in, _net_out = batch
        box_fmt = _map_pred_box_format(pl_module)
        img_tv, img_paths = _cpu_imgs_and_paths_from_net_in(
            net_in, tuple_use_tv_stack=False
        )
        B = len(metric_preds)

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]
            pred_xyxy, pred_cls, pred_score = metric_pred_item_to_xyxy_cls_score_numpy(
                metric_preds[i], box_fmt, self.conf_thres
            )

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            names = dataset.map_class_id_to_class_name or None
            panel_in = img_np.copy()
            panel_pred = ObjectDetectDataset.draw_label_on_numpy(
                img_np.copy(),
                pred_xyxy,
                pred_cls,
                class_names=names,
                colors=[(0, 0, 255)],
            )
            img_out = hstack_bgr_left_right_resize_right_to_match_left_with_gray_separator(
                panel_in, panel_pred
            )

            save_path = self.save_dir_pred / (img_path.stem + ".jpg")
            cv2.imwrite(str(save_path), img_out)

            self.csv_table_pred.append(
                {
                    "img_path": str(img_path),
                    "predictions": instances_to_json_str(
                        pred_xyxy, pred_cls, names, scores=pred_score
                    ),
                    "num_pred": int(pred_xyxy.shape[0]),
                    "save_path": str(save_path),
                }
            )

    def on_predict_epoch_start(self, trainer, pl_module):
        self.csv_table_pred = []
        return super().on_predict_epoch_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.save_dir is not None:
            write_row_dicts_to_csv_path_skip_if_empty(
                self.save_dir / "prediction_results.csv", self.csv_table_pred
            )


class LogObjectDetectVisualizationCallback(pl.Callback):
    """
    训练 / 验证每个 epoch **各记录一次**检测可视化到 TensorBoard（``train/sample_detections``、
    ``val/sample_detections``），均使用 **``batch_idx==0``** 的 batch。

    依赖 :class:`~lovely_deep_learning.module.object_detect.ObjectDetectModule`：``training_step``
    返回 ``{"loss", "metric_preds", "net_out"}``（``loss`` 供优化）；``validation_step`` 返回
    ``{"metric_preds", "net_out"}``（损失仅通过 ``self.log`` 记录）。

    默认 ``max_images=4``、``nrow=2``，拼成 **2×2** 正方形网格。
    """

    def __init__(
        self,
        max_images: int = 4,
        nrow: int = 2,
        match_iou_thres: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_images = int(max_images)
        self.nrow = int(nrow)
        self.match_iou_thres = float(match_iou_thres)

    def _log_sample_detections_tensorboard(
        self,
        pl_module: pl.LightningModule,
        batch: Any,
        outputs: dict[str, Any],
        dataset: ObjectDetectDataset,
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
        box_fmt = _map_pred_box_format(pl_module)

        n = min(self.max_images, len(metric_preds), len(net_in))
        if n == 0:
            return

        panels: list[torch.Tensor] = []
        for i in range(n):
            img_tensor = net_in[i]["img_tv_transformed"]
            img_np = dataset.convert_img_from_tensor_to_numpy(img_tensor)
            pr = metric_preds[i]
            pb = pr["boxes"].detach().cpu().float().numpy()
            ps = pr["scores"].detach().cpu().float().numpy()
            plb = pr["labels"].detach().cpu().numpy().astype(np.int64)
            pb_xyxy = boxes_layout_to_xyxy_numpy(pb, box_fmt)

            gb, gl = _gt_xyxy_cls_numpy(_net_out_sample(net_out, i))

            panel = ObjectDetectDataset.draw_target_and_predict_label_on_numpy(
                img_np,
                bboxes_pred=pb_xyxy,
                classes_pred=plb,
                bboxes_gt=gb,
                classes_gt=gl,
                class_names=dataset.map_class_id_to_class_name,
                pred_scores=ps,
                match_by_iou=True,
                iou_match_threshold=self.match_iou_thres,
            )
            t = ObjectDetectDataset.convert_img_from_numpy_to_tensor_uint8(panel)
            panels.append(t)

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
            self._log_sample_detections_tensorboard(
                pl_module, batch, outputs, dataset, tb_tag
            )
        except Exception as e:
            print(
                f"Warning: failed to log {phase_label} detection visualization at step "
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
            "train/sample_detections",
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
            "val/sample_detections",
            "val",
        )