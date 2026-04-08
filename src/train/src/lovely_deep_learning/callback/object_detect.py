import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

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


def _detections_tensor_to_numpy(
    det_i: torch.Tensor, conf_thres: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单张图 ``(max_det, 6)`` → xyxy、cls、score（已按 conf 过滤）。"""
    d = det_i.detach().cpu().float().numpy()
    if d.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    xywh = d[:, :4]
    score = d[:, 4]
    cls_id = d[:, 5].astype(np.int32)
    m = score >= conf_thres
    xywh, score, cls_id = xywh[m], score[m], cls_id[m]
    xyxy = _cxcywh_to_xyxy(xywh)
    return xyxy, cls_id, score


class ObjectDetectCallback(pl.Callback):
    """保存目标检测可视化。测试阶段使用 :meth:`ObjectDetectDataset.draw_target_and_predict_label_on_numpy`（左=预测，右=GT，IoU 配色）；预测阶段为左=原图、右=预测框。"""

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
        dataset: ObjectDetectDataset = trainer.test_dataloaders.dataset
        net_in, net_out = batch
        detections = outputs["detections"]
        if isinstance(net_in, dict):
            img_tv = net_in["img_tv_transformed"]
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
        B = detections.shape[0]

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]
            det_i = detections[i]

            pred_xyxy, pred_cls, pred_score = _detections_tensor_to_numpy(
                det_i, self.conf_thres
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
        dataset: ObjectDetectDataset = trainer.predict_dataloaders.dataset
        net_in, _ = batch
        detections = outputs["detections"]

        if isinstance(net_in, dict):
            img_tv = net_in["img_tv_transformed"]
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
        B = detections.shape[0]

        for i in range(B):
            img_path = Path(img_paths[i])
            img = img_tv[i]
            det_i = detections[i]
            pred_xyxy, pred_cls, pred_score = _detections_tensor_to_numpy(
                det_i, self.conf_thres
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
