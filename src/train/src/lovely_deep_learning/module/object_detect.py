"""
基于 ``DAGNet`` + ``configs/models/yolov8_n.yaml`` 的目标检测 Lightning 模块；
损失为 :class:`~lovely_deep_learning.loss.object_detect.DetectionLossYOLOv8`（
自研前向，与 Ultralytics ``v8DetectionLoss`` 公式对齐；便于对照源码与论文）。

数据侧使用 ``lovely_deep_learning.data_module.object_detect.ObjectDetectDataModule``。
实验 YAML 中请将 ``model.class_path`` 指向本模块的 ``ObjectDetectModule``。
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import instantiate_class
from typing import Any

from lovely_deep_learning.model.DAGNet import DAGNet

from ..dataset.object_detect import postprocess_detections
from ..loss.object_detect import DetectionLossYOLOv8
from ..metric.object_detect import ObjectDetectMetric

class ObjectDetectModule(pl.LightningModule):
    def __init__(
        self,
        model: Any = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        criterion: Any = None,
        metrics: Any = None,
    ):
        super().__init__()
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        if self.optimizer_cfg is None:
            raise ValueError(
                "`optimizer` config is required in YAML (model.init_args.optimizer).")
        if self.lr_scheduler_cfg is None:
            raise ValueError(
                "`lr_scheduler` config is required in YAML (model.init_args.lr_scheduler).")
        if criterion is None:
            raise ValueError(
                "`criterion` config is required in YAML (model.init_args.criterion).")
        if metrics is None:
            raise ValueError(
                "`metrics` config is required in YAML (model.init_args.metrics).")
        if model is None:
            raise ValueError(
                "`model` config is required (DAGNet / yolov8_n.yaml 字段).")

        self.model = DAGNet(**model)
        self.criterion: DetectionLossYOLOv8 = criterion
        self.metrics: ObjectDetectMetric = metrics

        last_name = self.model.layers_config[-1]["name"]
        self._detect = self.model.layers[last_name]


    def forward(self, x: torch.Tensor):
        return self.model([x])

    def training_step(self, batch, batch_idx):
        # batch: (net_in, net_out)，长度均为 B（batch size）；与 DataLoader 中样本顺序一致。
        net_in, net_out = batch
        # imgs: (B, C, H, W)，与 net_in[i]["img"] 单张 (C, H, W) 沿 batch 维 stack。
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        preds = self.model([imgs])
        loss = self.criterion(preds, net_out=net_out)
        bs = imgs.shape[0]  # B

        # 训练态 Detect.forward 返回多尺度特征 list；可直接调用 Detect._inference 解码，
        # 避免为记录 mAP 再做一次 eval 前向。
        train_feats = self._unpack_raw_detection_output(preds)
        with torch.no_grad():
            raw = self._detect._inference([t.detach() for t in train_feats])
            nms, nms_iou, conf_thres = self._get_postprocess_cfg()
            detections = postprocess_detections(
                raw=raw,
                max_det=self._detect.max_det,
                nc=self._detect.nc,
                nms=nms,
                conf_thres=conf_thres,
                nms_iou=nms_iou,
            )
            preds_map, targets_map = self._build_map_inputs(detections, net_out)
            self.metrics.update("train", preds_map, targets_map)

        self.log(
            "train_loss",
            loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        metrics = self.metrics.compute("train")
        self.log("train_map", metrics["map"], prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("train")

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        preds_for_loss = self.model([imgs])
        loss = self.criterion(preds_for_loss, net_out=net_out)
        bs = imgs.shape[0]

        self.log(
            "val_loss",
            loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # val 下 Lightning 已 eval；与 loss 共用前向，``raw`` 为 Detect 解码输出（再经 top-k / NMS）。
        raw = self._unpack_raw_detection_output(preds_for_loss)
        nms, nms_iou, conf_thres = self._get_postprocess_cfg()
        detections = postprocess_detections(
            raw=raw,
            max_det=self._detect.max_det,
            nc=self._detect.nc,
            nms=nms,
            conf_thres=conf_thres,
            nms_iou=nms_iou,
        )
        preds, targets = self._build_map_inputs(detections, net_out)
        self.metrics.update("val", preds, targets)

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        preds_for_loss = self.model([imgs])
        loss = self.criterion(preds_for_loss, net_out=net_out)
        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        raw = self._unpack_raw_detection_output(preds_for_loss)
        nms, nms_iou, conf_thres = self._get_postprocess_cfg()
        detections = postprocess_detections(
            raw=raw,
            max_det=self._detect.max_det,
            nc=self._detect.nc,
            nms=nms,
            conf_thres=conf_thres,
            nms_iou=nms_iou,
        )
        preds, targets = self._build_map_inputs(detections, net_out)
        self.metrics.update("test", preds, targets)
        return {"detections": detections}

    def predict_step(self, batch, batch_idx):
        net_in, _net_out = batch
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        self.eval()
        with torch.no_grad():
            out = self.model([imgs])
            raw = self._unpack_raw_detection_output(out)
            nms, nms_iou, conf_thres = self._get_postprocess_cfg()
            detections = postprocess_detections(
                raw=raw,
                max_det=self._detect.max_det,
                nc=self._detect.nc,
                nms=nms,
                conf_thres=conf_thres,
                nms_iou=nms_iou,
            )
        return {"detections": detections}

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute("val")
        self.log("val_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("val")

    def on_test_epoch_end(self):
        metrics = self.metrics.compute("test")
        self.log("test_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("test")

    def configure_optimizers(self):
        optimizer = instantiate_class(
            self.model.parameters(), self.optimizer_cfg)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # -----------------------------
    # Inference / postprocess
    # -----------------------------

    @staticmethod
    def _unpack_raw_detection_output(dag_out: tuple) -> torch.Tensor:
        """DAGNet 返回 ``(最后一层输出,)``；Detect 在 eval 下返回 ``(raw_pred, feats)``。"""
        layer_out = dag_out[0]
        if isinstance(layer_out, tuple):
            return layer_out[0]
        return layer_out

    @staticmethod
    def _cxcywh_pixels_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """``boxes``: ``(..., 4)`` 中心 cxcywh 像素 → xyxy。"""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack(
            (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
        )

    # -----------------------------
    # Loss / metric input builders
    # -----------------------------
    def _build_map_inputs(
        self, detections: torch.Tensor, net_out: Any
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        preds: list[dict[str, torch.Tensor]] = []
        targets: list[dict[str, torch.Tensor]] = []
        _nms, _nms_iou, conf_thres = self._get_postprocess_cfg()

        if isinstance(net_out, dict):
            batch_size = detections.shape[0]
            net_out_list = (
                [{k: net_out[k][i] for k in net_out}
                    for i in range(batch_size)]
                if net_out
                else [{} for _ in range(batch_size)]
            )
        else:
            net_out_list = list(net_out)

        for det_row, gt in zip(detections, net_out_list):
            pred_mask = det_row[:, 4] > conf_thres
            pred_row = det_row[pred_mask]
            if pred_row.numel() == 0:
                pred_boxes = det_row.new_zeros((0, 4))
                pred_scores = det_row.new_zeros((0,))
                pred_labels = torch.zeros(
                    (0,), device=det_row.device, dtype=torch.long)
            else:
                pred_boxes = self._cxcywh_pixels_to_xyxy(
                    pred_row[:, :4]).float()
                pred_scores = pred_row[:, 4].float()
                pred_labels = pred_row[:, 5].long()

            preds.append(
                {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
            )

            if gt and "bboxes_xyxy_abs_tv_transformed" in gt:
                gt_boxes = gt["bboxes_xyxy_abs_tv_transformed"]
                if hasattr(gt_boxes, "data"):
                    gt_boxes = gt_boxes.data
                elif hasattr(gt_boxes, "as_tensor"):
                    gt_boxes = gt_boxes.as_tensor()
                gt_boxes = gt_boxes.to(
                    device=det_row.device, dtype=torch.float32)
                gt_labels = gt["cls_tv_transformed"].to(
                    device=det_row.device).long().reshape(-1)
            else:
                gt_boxes = det_row.new_zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros(
                    (0,), device=det_row.device, dtype=torch.long)

            targets.append({"boxes": gt_boxes, "labels": gt_labels})

        return preds, targets

    def _get_postprocess_cfg(self) -> tuple[bool, float, float]:
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is None:
            return True, 0.7, 0.001
        for name in ("train_dataset", "val_dataset", "test_dataset", "pred_dataset"):
            ds = getattr(dm, name, None)
            if ds is not None:
                return bool(ds.nms), float(ds.nms_iou), float(ds.inference_conf_thres)
        return True, 0.7, 0.001
