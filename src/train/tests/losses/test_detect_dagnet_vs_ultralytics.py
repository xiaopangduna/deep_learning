"""
对照 Ultralytics YOLOv8 的工具函数与 ``v8DetectionLoss``，验证 ``detect_dagnet_loss`` 中
几何解码与 **自研** ``DetectDAGNetLoss`` 是否与官方前向 **逐分量** 完全一致。

说明
----
- **CIoU、make_anchors、DFL 期望 + dist2bbox**：与 ``ultralytics.utils`` 对齐（``allclose``）。
- **总损失**：``DetectDAGNetLoss.forward_loss_vec`` 应与 ``v8DetectionLoss`` 返回的第一个 ``(3,)``
  张量 **完全一致**（同一 ``preds``、``batch``、超参）；对照测试使用 ``datasets/COCO8`` 真实标注。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T
import yaml

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox as u_dist2bbox
from ultralytics.utils.tal import make_anchors as u_make_anchors

from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset
from lovely_deep_learning.losses.detect_dagnet_loss import (
    DetectDAGNetLoss,
    _bbox_ciou,
    _dist2bbox,
    _dfl_decode,
    _make_anchors,
    _merge_hyp_args,
    _V8LossAdapter,
)
from lovely_deep_learning.model.DAGNet import DAGNet
from lovely_deep_learning.module.object_detect import ObjectDetectModule

# 与仓库内实验配置一致，便于本地跑 pytest
_YOLO_PT = Path(__file__).resolve().parents[2] / "pretrained_models" / "yolov8n.pt"
_YAML = Path(__file__).resolve().parents[2] / "configs" / "models" / "yolov8_n.yaml"
_COCO_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "COCO8"
_COCO_TRAIN_CSV = _COCO_ROOT / "train.csv"
_COCO_MAP = _COCO_ROOT / "map_class_id_to_class_name.csv"


def _collate_object_detect_batch(
    net_in_list: list, net_out_list: list
) -> dict[str, torch.Tensor]:
    """与 ``ObjectDetectModule._collate_to_target_batch`` 列表分支一致（供测试构造 batch）。"""
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
        xywh = ObjectDetectModule._xyxy_abs_pixels_to_xywh_norm(
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


def test_ciou_matches_ultralytics_bbox_iou_xyxy():
    """``_bbox_ciou`` 与 ``bbox_iou(..., xywh=False, CIoU=True)`` 在 xyxy 上应对齐。"""
    torch.manual_seed(0)
    box1 = torch.rand(32, 4) * 400
    box2 = torch.rand(32, 4) * 400
    for b in (box1, box2):
        b[:, 2] = b[:, 0] + b[:, 2].abs() + 1
        b[:, 3] = b[:, 1] + b[:, 3].abs() + 1

    ours = _bbox_ciou(box1, box2)
    ref = bbox_iou(box1, box2, xywh=False, CIoU=True).view(-1)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)


def test_make_anchors_matches_ultralytics():
    """三层假特征图上的锚点与 stride 与 ``tal.make_anchors`` 一致。"""
    torch.manual_seed(1)
    b = 2
    feats = [
        torch.randn(b, 64, 80, 80),
        torch.randn(b, 128, 40, 40),
        torch.randn(b, 256, 20, 20),
    ]
    strides = torch.tensor([8.0, 16.0, 32.0])

    a1, s1 = _make_anchors(feats, strides)
    a2, s2 = u_make_anchors(feats, strides, grid_cell_offset=0.5)
    assert torch.allclose(a1, a2, atol=0.0, rtol=0.0)
    assert torch.allclose(s1, s2, atol=0.0, rtol=0.0)


def test_dfl_decode_and_dist2bbox_match_ultralytics():
    """DFL softmax×proj + ``dist2bbox(xywh=False)`` 与官方解码路径一致（网格坐标系）。"""
    from ultralytics import YOLO

    if not _YOLO_PT.is_file():
        pytest.skip(f"missing weights: {_YOLO_PT}")

    m = YOLO(str(_YOLO_PT)).model
    det = m.model[-1]
    reg_max = int(det.reg_max)
    b = 1
    a = 80 * 80 + 40 * 40 + 20 * 20
    device = torch.device("cpu")
    pred_distri = torch.randn(b, a, 4 * reg_max, device=device)
    strides = det.stride.to(device)
    feats = [
        torch.zeros(b, 64, 80, 80, device=device),
        torch.zeros(b, 128, 40, 40, device=device),
        torch.zeros(b, 256, 20, 20, device=device),
    ]
    ap_u, st_u = u_make_anchors(feats, strides, 0.5)
    ap_u = ap_u.unsqueeze(0).expand(b, -1, -1)

    proj = torch.arange(reg_max, dtype=pred_distri.dtype, device=device)
    pd = pred_distri.view(b, a, 4, reg_max).softmax(-1).matmul(proj)
    pred_u = u_dist2bbox(pd, ap_u, xywh=False)

    pd_mine = _dfl_decode(pred_distri, reg_max)
    pred_m = _dist2bbox(pd_mine, ap_u)

    assert torch.allclose(pred_m, pred_u, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not _YOLO_PT.is_file(), reason="pretrained yolov8n.pt not found")
def test_dagnet_loss_finite_on_random_batch():
    """``DetectDAGNetLoss`` 在随机 batch 上可前向、可反传。"""
    with open(_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    net = DAGNet(**cfg)
    net.train()
    det = net.layers[net.layers_config[-1]["name"]]
    crit = DetectDAGNetLoss(nc=det.nc, reg_max=det.reg_max, stride=det.stride)
    b, h, w = 2, 640, 640
    img = torch.rand(b, 3, h, w)
    preds = net([img])[0]
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([0.0, 3.0, 5.0]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.05, 0.05], [0.4, 0.4, 0.1, 0.2]]
        ),
    }
    loss = crit(preds, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.skipif(not _YOLO_PT.is_file(), reason="pretrained yolov8n.pt not found")
def test_ultralytics_v8_loss_finite_same_batch():
    """官方 ``v8DetectionLoss`` 在同一 ``batch`` 上可运行（供与上面对照，数值不要求相等）。"""
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss

    m = YOLO(str(_YOLO_PT)).model
    m.train()
    m.args = DEFAULT_CFG
    crit = v8DetectionLoss(m)

    b, h, w = 2, 640, 640
    img = torch.rand(b, 3, h, w)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([0.0, 3.0, 5.0]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.05, 0.05], [0.4, 0.4, 0.1, 0.2]]
        ),
    }
    loss_vec, _ = m(batch)
    assert loss_vec.shape == (3,)
    assert torch.isfinite(loss_vec).all()
    preds = m.forward(batch["img"])
    loss_vec2, _ = crit(preds, batch)
    assert torch.allclose(loss_vec, loss_vec2)


@pytest.mark.skipif(not _YOLO_PT.is_file(), reason="pretrained yolov8n.pt not found")
@pytest.mark.skipif(not _COCO_TRAIN_CSV.is_file(), reason="datasets/COCO8/train.csv not found")
@pytest.mark.skipif(not _COCO_MAP.is_file(), reason="COCO8 class map csv not found")
def test_detect_dagnet_loss_matches_v8_detection_loss_coco8():
    """
    自研 ``DetectDAGNetLoss`` 与 ``v8DetectionLoss`` 在 **同一 preds、同一 batch** 下
    三个分量（已含 gain 与 batch_size）应与官方 **逐元素一致**；数据来自 ``datasets/COCO8``。
    """
    from ultralytics.utils.loss import v8DetectionLoss

    torch.manual_seed(0)
    with open(_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dag = DAGNet(**cfg)
    dag.train()

    tf = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = ObjectDetectDataset(
        [_COCO_TRAIN_CSV],
        key_map={"img_path": "path_img", "object_label_path": "path_label_detect_yolo"},
        transform=tf,
        map_class_id_to_class_name=_COCO_MAP,
    )
    assert len(ds) >= 2, "COCO8 train.csv should have at least 2 samples"
    s0, s1 = ds[0], ds[1]
    batch = _collate_object_detect_batch([s0[0], s1[0]], [s0[1], s1[1]])
    img = batch["img"]
    preds = dag([img])[0]

    hyp = _merge_hyp_args(7.5, 0.5, 1.5)
    det = dag.layers[dag.layers_config[-1]["name"]]
    crit_ours = DetectDAGNetLoss(
        nc=det.nc,
        reg_max=det.reg_max,
        stride=det.stride,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10,
    )
    loss_vec_ours = crit_ours.forward_loss_vec(preds, batch)

    adapter = _V8LossAdapter(dag, hyp)
    crit_v8 = v8DetectionLoss(adapter, tal_topk=10)
    loss_vec_v8, _ = crit_v8(preds, batch)

    assert loss_vec_ours.shape == (3,)
    assert torch.allclose(loss_vec_ours, loss_vec_v8, atol=0.0, rtol=0.0)
    assert torch.allclose(loss_vec_ours.sum(), loss_vec_v8.sum(), atol=0.0, rtol=0.0)
