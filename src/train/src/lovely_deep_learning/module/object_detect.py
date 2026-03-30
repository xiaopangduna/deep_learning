"""
基于 ``DAGNet`` + ``configs/models/yolov8_n.yaml`` 的目标检测 Lightning 模块；
损失为 :class:`~lovely_deep_learning.losses.detect_dagnet_loss.DetectDAGNetLoss`（
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
from lovely_deep_learning.losses.detect_dagnet_loss import DetectDAGNetLoss

from ..dataset.object_detect import ObjectDetectDataset


class ObjectDetectModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model=None,
        init_type: str | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        if self.optimizer_cfg is None:
            raise ValueError("`optimizer` config is required in YAML (model.init_args.optimizer).")
        if self.lr_scheduler_cfg is None:
            raise ValueError("`lr_scheduler` config is required in YAML (model.init_args.lr_scheduler).")
        if model is None:
            raise ValueError("`model` config is required (DAGNet / yolov8_n.yaml 字段).")

        self.model = DAGNet(**model)
        last_name = self.model.layers_config[-1]["name"]
        self._detect = self.model.layers[last_name]
        loss_kw = dict(loss or {})
        self.criterion = DetectDAGNetLoss(
            nc=self._detect.nc,
            reg_max=self._detect.reg_max,
            stride=self._detect.stride,
            **loss_kw,
        )

        self._graph_logged = False

    def forward(self, x: torch.Tensor):
        return self.model([x])

    @staticmethod
    def _xyxy_abs_pixels_to_xywh_norm(
        xyxy: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        x1 = xyxy[..., 0]
        y1 = xyxy[..., 1]
        x2 = xyxy[..., 2]
        y2 = xyxy[..., 3]
        w = float(width)
        h = float(height)
        cx = (x1 + x2) * 0.5 / w
        cy = (y1 + y2) * 0.5 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return torch.stack((cx, cy, bw, bh), dim=-1)

    def _collate_to_target_batch(self, net_in: Any, net_out: Any) -> dict[str, torch.Tensor]:
        if isinstance(net_in, dict):
            img_batch = net_in["img_tv_transformed"]
            if hasattr(img_batch, "data"):
                img_batch = img_batch.data
            batch_size = img_batch.shape[0]
            net_in_list = [{k: net_in[k][i] for k in net_in} for i in range(batch_size)]
            net_out_list = (
                [{k: net_out[k][i] for k in net_out} for i in range(batch_size)]
                if isinstance(net_out, dict) and net_out
                else [{} for _ in range(batch_size)]
            )
        else:
            net_in_list = list(net_in)
            net_out_list = list(net_out)
            img_batch = torch.stack(
                [
                    ni["img_tv_transformed"].data
                    if hasattr(ni["img_tv_transformed"], "data")
                    else ni["img_tv_transformed"]
                    for ni in net_in_list
                ],
                dim=0,
            )

        device = img_batch.device
        batch_idx_parts: list[torch.Tensor] = []
        cls_parts: list[torch.Tensor] = []
        bbox_parts: list[torch.Tensor] = []

        for i, (ni, no) in enumerate(zip(net_in_list, net_out_list)):
            _c, h_i, w_i = img_batch[i].shape
            if not no:
                continue
            cls_t = no["cls_tv_transformed"]
            boxes = no["bboxes_xyxy_abs_tv_transformed"]
            if hasattr(boxes, "data"):
                boxes = boxes.data
            elif hasattr(boxes, "as_tensor"):
                boxes = boxes.as_tensor()
            n = int(cls_t.shape[0])
            if n == 0:
                continue
            xywh = self._xyxy_abs_pixels_to_xywh_norm(
                boxes.float(), int(h_i), int(w_i)
            ).to(device)
            batch_idx_parts.append(
                torch.full((n,), float(i), device=device, dtype=torch.float32)
            )
            cls_parts.append(cls_t.float().to(device).reshape(-1))
            bbox_parts.append(xywh)

        if not cls_parts:
            z = img_batch.new_zeros((0,), dtype=torch.float32)
            z4 = img_batch.new_zeros((0, 4), dtype=torch.float32)
            return {"img": img_batch, "batch_idx": z, "cls": z, "bboxes": z4}

        return {
            "img": img_batch,
            "batch_idx": torch.cat(batch_idx_parts, dim=0),
            "cls": torch.cat(cls_parts, dim=0),
            "bboxes": torch.cat(bbox_parts, dim=0),
        }

    def _forward_train_preds(self, imgs: torch.Tensor):
        out = self.model([imgs])
        if isinstance(out, tuple) and len(out) == 1:
            return out[0]
        return out

    def _compute_loss(self, tb: dict) -> torch.Tensor:
        prev = self.training
        self.train()
        try:
            preds = self._forward_train_preds(tb["img"])
            return self.criterion(
                preds,
                batch_idx=tb["batch_idx"],
                cls=tb["cls"],
                bboxes=tb["bboxes"],
            )
        finally:
            if not prev:
                self.eval()

    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        tb = self._collate_to_target_batch(net_in, net_out)
        loss = self._compute_loss(tb)
        bs = tb["img"].shape[0]

        if batch_idx == 0:
            try:
                dataset: ObjectDetectDataset = self.trainer.datamodule.train_dataset
                self._log_detection_images(net_in, net_out, dataset, "train")
            except Exception as e:
                print(f"Warning: failed to log detection images at step {self.global_step}, {e}")

        self.log(
            "train_loss",
            loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        tb = self._collate_to_target_batch(net_in, net_out)
        loss = self._compute_loss(tb)
        bs = tb["img"].shape[0]

        if batch_idx == 0:
            try:
                dataset: ObjectDetectDataset = self.trainer.datamodule.val_dataset
                self._log_detection_images(net_in, net_out, dataset, "val")
            except Exception as e:
                print(f"Warning: failed to log detection images at step {self.global_step}, {e}")

        self.log(
            "val_loss",
            loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        tb = self._collate_to_target_batch(net_in, net_out)
        loss = self._compute_loss(tb)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        net_in, _ = batch
        if isinstance(net_in, dict):
            imgs = net_in["img_tv_transformed"]
            if hasattr(imgs, "data"):
                imgs = imgs.data
        else:
            imgs = torch.stack(
                [
                    ni["img_tv_transformed"].data
                    if hasattr(ni["img_tv_transformed"], "data")
                    else ni["img_tv_transformed"]
                    for ni in net_in
                ],
                dim=0,
            )
        self.eval()
        with torch.no_grad():
            return self.forward(imgs)

    def configure_optimizers(self):
        optimizer = instantiate_class(self.model.parameters(), self.optimizer_cfg)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_detection_images(self, net_in, net_out, dataset: ObjectDetectDataset, log_prefix: str):
        if self.logger is None:
            return
        from torchvision.utils import make_grid

        if isinstance(net_in, dict):
            n_show = min(4, net_in["img_tv_transformed"].shape[0])
            rows = []
            for i in range(n_show):
                ni = {k: net_in[k][i] for k in net_in}
                no = (
                    {k: net_out[k][i] for k in net_out}
                    if isinstance(net_out, dict) and net_out
                    else {}
                )
                rows.append(self._one_sample_to_viz_tensor(ni, no, dataset))
        else:
            n_show = min(4, len(net_in))
            rows = []
            for i in range(n_show):
                rows.append(self._one_sample_to_viz_tensor(net_in[i], net_out[i], dataset))

        if not rows:
            return
        grid = make_grid(rows, nrow=min(2, len(rows)))
        self.logger.experiment.add_image(
            f"{log_prefix}/sample_batch", grid, global_step=self.global_step
        )

    def _one_sample_to_viz_tensor(self, ni: dict, no: dict, dataset: ObjectDetectDataset):
        from ..dataset.base import BaseDataset

        img_t = ni["img_tv_transformed"]
        if hasattr(img_t, "data"):
            img_t = img_t.data
        img_np = dataset.convert_img_from_tensor_to_numpy(img_t)
        if no and "bboxes_xyxy_abs_tv_transformed" in no:
            boxes = no["bboxes_xyxy_abs_tv_transformed"]
            if hasattr(boxes, "data"):
                boxes = boxes.data.cpu().numpy()
            else:
                boxes = boxes.cpu().numpy()
            cls_ids = no["cls_tv_transformed"].cpu().numpy()
            names = dataset.map_class_id_to_class_name or None
            img_np = ObjectDetectDataset.draw_label_on_numpy(
                img_np, boxes, cls_ids, class_names=names
            )

        t = BaseDataset.convert_img_from_numpy_to_tensor_uint8(img_np)
        return t.float() / 255.0
