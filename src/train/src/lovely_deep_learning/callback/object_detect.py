import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from ..dataset.object_detect import (
    ObjectDetectDataset,
    greedy_match_pred_gt_iou,
)


def _class_name_lookup(names: Mapping[Any, str] | None, class_id: int) -> str | None:
    if names is None:
        return None
    return names.get(class_id, names.get(str(class_id)))


def _instances_to_json_str(
    xyxy: np.ndarray,
    cls_ids: np.ndarray,
    class_names: Mapping[Any, str] | None,
    scores: np.ndarray | None = None,
) -> str:
    """每张图的目标列表，写入 CSV 的一列（JSON 文本）。"""
    rows: list[dict[str, Any]] = []
    for j in range(xyxy.shape[0]):
        cid = int(cls_ids[j])
        item: dict[str, Any] = {
            "class_id": cid,
            "bbox_xyxy": [float(xyxy[j, k]) for k in range(4)],
        }
        cname = _class_name_lookup(class_names, cid)
        if cname is not None:
            item["class_name"] = cname
        if scores is not None and j < len(scores):
            item["confidence"] = float(scores[j])
        rows.append(item)
    return json.dumps(rows, ensure_ascii=False)


def _side_by_side_bgr(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """左右拼接两张同高 BGR 图；宽度不一致时将右侧缩放到与左侧同尺寸。"""
    h, wl = left_bgr.shape[:2]
    hr, wr = right_bgr.shape[:2]
    if (h, wl) != (hr, wr):
        right_bgr = cv2.resize(right_bgr, (wl, h), interpolation=cv2.INTER_LINEAR)
    gap_w = 3
    sep = np.full((h, gap_w, 3), 220, dtype=np.uint8)
    return np.hstack([left_bgr, sep, right_bgr])


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _boxes_layout_to_xyxy_numpy(boxes: np.ndarray, box_format: str) -> np.ndarray:
    """``postprocess.run`` 的 ``boxes`` ``(N,4)``（由 ``map_pred_box_format`` 约定）→ 像素 xyxy。"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    bf = box_format.lower()
    b = boxes.reshape(-1, 4).astype(np.float32)
    if bf == "xyxy":
        return b
    if bf == "cxcywh":
        return _cxcywh_to_xyxy(b)
    if bf == "xywh":
        x1, y1, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([x1, y1, x1 + w, y1 + h], axis=1).astype(np.float32)
    return b


def _map_pred_item_to_pred_xyxy_cls_score(
    pr: dict[str, Any],
    box_format: str,
    conf_thres: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    单张图的 ``map_preds[i]`` → 像素 xyxy、类别、分数；可选再按 ``conf_thres`` 过滤。
    """
    pb = pr["boxes"].detach().cpu().float().numpy()
    ps = pr["scores"].detach().cpu().float().numpy()
    plb = pr["labels"].detach().cpu().numpy().astype(np.int32)
    if conf_thres is not None:
        m = ps >= float(conf_thres)
        pb, ps, plb = pb[m], ps[m], plb[m]
    pb_xyxy = _boxes_layout_to_xyxy_numpy(pb, box_format)
    return pb_xyxy, plb, ps


def _stack_visual_imgs_from_net_in(net_in: Any) -> torch.Tensor:
    """test/val 用 ``img_tv_transformed``；predict 可能仅有 ``img``。"""
    if isinstance(net_in, dict):
        key = "img_tv_transformed" if "img_tv_transformed" in net_in else "img"
        return net_in[key]
    first = net_in[0]
    key = "img_tv_transformed" if "img_tv_transformed" in first else "img"
    return torch.stack([ni[key] for ni in net_in], dim=0)


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
        map_preds = outputs.get("map_preds")
        if map_preds is None:
            return
        dataset: ObjectDetectDataset = trainer.test_dataloaders.dataset
        net_in, net_out = batch
        post = getattr(pl_module, "postprocess", None)
        box_fmt = str(post.map_pred_box_format) if post else "xyxy"

        if isinstance(net_in, dict):
            img_tv = net_in.get("img_tv_transformed", net_in.get("img"))
            img_paths = net_in["img_path"]
        else:
            img_tv = torch.stack(
                [
                    ni["img_tv_transformed"].data
                    if hasattr(ni["img_tv_transformed"], "data")
                    else ni["img_tv_transformed"]
                    for ni in net_in
                ],
                dim=0,
            )
            img_paths = [ni["img_path"] for ni in net_in]

        img_tv = img_tv.detach().cpu()
        B = len(map_preds)

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]

            pred_xyxy, pred_cls, pred_score = _map_pred_item_to_pred_xyxy_cls_score(
                map_preds[i], box_fmt, self.conf_thres
            )

            if isinstance(net_out, dict):
                no_i = {k: net_out[k][i] for k in net_out}
            else:
                no_i = net_out[i]

            if no_i and "bboxes_xyxy_abs_tv_transformed" in no_i:
                boxes = no_i["bboxes_xyxy_abs_tv_transformed"]
                if hasattr(boxes, "data"):
                    boxes = boxes.data.cpu().numpy()
                else:
                    boxes = boxes.cpu().numpy()
                cls_ids = no_i["cls_tv_transformed"].cpu().numpy().reshape(-1)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                cls_ids = np.zeros((0,), dtype=np.int32)

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
                    "ground_truth": _instances_to_json_str(
                        boxes.astype(np.float32),
                        cls_ids.astype(np.int32),
                        names,
                        scores=None,
                    ),
                    "predictions": _instances_to_json_str(
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
        if self.save_dir is not None and self.csv_table_test:
            df = pd.DataFrame(self.csv_table_test)
            csv_save_path = self.save_dir / "test_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"检测结果已保存到: {csv_save_path}")

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
        map_preds = outputs.get("map_preds")
        if map_preds is None:
            return
        dataset: ObjectDetectDataset = trainer.predict_dataloaders.dataset
        net_in, _net_out = batch
        post = getattr(pl_module, "postprocess", None)
        box_fmt = str(post.map_pred_box_format) if post else "xyxy"

        if isinstance(net_in, dict):
            img_tv = net_in.get("img_tv_transformed", net_in.get("img"))
            img_paths = net_in["img_path"]
        else:
            img_tv = _stack_visual_imgs_from_net_in(net_in)
            img_paths = [ni["img_path"] for ni in net_in]

        img_tv = img_tv.detach().cpu()
        B = len(map_preds)

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]
            pred_xyxy, pred_cls, pred_score = _map_pred_item_to_pred_xyxy_cls_score(
                map_preds[i], box_fmt, self.conf_thres
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
            img_out = _side_by_side_bgr(panel_in, panel_pred)

            save_path = self.save_dir_pred / (img_path.stem + ".jpg")
            cv2.imwrite(str(save_path), img_out)

            self.csv_table_pred.append(
                {
                    "img_path": str(img_path),
                    "predictions": _instances_to_json_str(
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
        if self.save_dir is not None and self.csv_table_pred:
            df = pd.DataFrame(self.csv_table_pred)
            csv_save_path = self.save_dir / "prediction_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"检测结果已保存到: {csv_save_path}")


class LogObjectDetectVisualizationCallback(pl.Callback):
    """
    训练 / 验证每个 epoch **各记录一次**检测可视化到 TensorBoard（``train/sample_detections``、
    ``val/sample_detections``），均使用 **``batch_idx==0``** 的 batch。

    依赖 :class:`~lovely_deep_learning.module.object_detect.ObjectDetectModule`：``training_step``
    返回 ``{"loss", "map_preds", "net_out"}``（``loss`` 供优化）；``validation_step`` 返回
    ``{"map_preds", "net_out"}``（损失仅通过 ``self.log`` 记录）。

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

    @staticmethod
    def _net_out_i(net_out: Any, i: int) -> dict[str, Any]:
        if isinstance(net_out, dict):
            return {k: net_out[k][i] for k in net_out}
        return net_out[i]

    @staticmethod
    def _gt_xyxy_cls_numpy(gt: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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

    def _pred_boxes_to_xyxy_numpy(
        self, boxes: np.ndarray, box_format: str
    ) -> np.ndarray:
        return _boxes_layout_to_xyxy_numpy(boxes, box_format)

    def _log_sample_detections_tensorboard(
        self,
        pl_module: pl.LightningModule,
        batch: Any,
        outputs: dict[str, Any],
        dataset: ObjectDetectDataset,
        tb_tag: str,
    ) -> None:
        map_preds = outputs.get("map_preds")
        net_out = outputs.get("net_out")
        if map_preds is None or net_out is None:
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
        box_fmt = "xyxy"
        post = getattr(pl_module, "postprocess", None)
        if post is not None and hasattr(post, "map_pred_box_format"):
            box_fmt = str(post.map_pred_box_format)

        n = min(self.max_images, len(map_preds), len(net_in))
        if n == 0:
            return

        panels: list[torch.Tensor] = []
        for i in range(n):
            img_tensor = net_in[i]["img_tv_transformed"]
            img_np = dataset.convert_img_from_tensor_to_numpy(img_tensor)
            pr = map_preds[i]
            pb = pr["boxes"].detach().cpu().float().numpy()
            ps = pr["scores"].detach().cpu().float().numpy()
            plb = pr["labels"].detach().cpu().numpy().astype(np.int64)
            pb_xyxy = self._pred_boxes_to_xyxy_numpy(pb, box_fmt)

            gt_dict = self._net_out_i(net_out, i)
            gb, gl = self._gt_xyxy_cls_numpy(gt_dict)

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