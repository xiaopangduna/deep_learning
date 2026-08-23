from __future__ import annotations

import torch
from ultralytics.utils.ops import non_max_suppression

from lovely_deep_learning.dataset.object_detect import (
    apply_nms_to_detections,
    cxcywh_pixels_to_xyxy,
    postprocess_detections,
)
from lovely_deep_learning.nn.head import Detect


def _rand_raw(batch: int = 2, nc: int = 80, anchors: int = 840, seed: int = 0) -> torch.Tensor:
    """随机 ``(B, 4+nc, A)``：像素 cxcywh + (0,1) 类分。"""
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(batch, 4 + nc, anchors, generator=g)
    raw[:, 0] = raw[:, 0] * 640.0
    raw[:, 1] = raw[:, 1] * 640.0
    raw[:, 2] = raw[:, 2] * 40.0 + 8.0
    raw[:, 3] = raw[:, 3] * 40.0 + 8.0
    return raw


def _valid_rows(dets: torch.Tensor) -> torch.Tensor:
    return dets[dets[:, 4] > 0]


def test_postprocess_detections_matches_official_nms():
    raw = _rand_raw()
    raw_before = raw.clone()
    max_det, nc = 300, 80
    conf, iou = 0.001, 0.7
    dets = postprocess_detections(
        raw,
        max_det=max_det,
        nc=nc,
        nms=True,
        conf_thres=conf,
        nms_iou=iou,
        multi_label=True,
    )
    assert torch.equal(raw, raw_before)
    assert dets.shape == (raw.shape[0], max_det, 6)

    official = non_max_suppression(
        raw.clone(),
        conf_thres=conf,
        iou_thres=iou,
        nc=nc,
        multi_label=True,
        max_det=max_det,
        in_place=False,
    )
    for b, pred in enumerate(official):
        n = 0 if pred is None else int(pred.shape[0])
        if n == 0:
            assert torch.count_nonzero(dets[b]) == 0
            continue
        got = dets[b, :n]
        assert torch.count_nonzero(dets[b, n:]) == 0
        assert torch.allclose(cxcywh_pixels_to_xyxy(got[:, :4]), pred[:, :4], atol=1e-5, rtol=1e-5)
        assert torch.allclose(got[:, 4], pred[:, 4], atol=1e-5, rtol=1e-5)
        assert torch.allclose(got[:, 5], pred[:, 5], atol=1e-5, rtol=1e-5)


def test_multi_label_keeps_two_classes_on_same_box():
    nc, anchors = 4, 5
    raw = torch.zeros(1, 4 + nc, anchors)
    raw[0, 0, 0] = 100.0
    raw[0, 1, 0] = 100.0
    raw[0, 2, 0] = 20.0
    raw[0, 3, 0] = 20.0
    raw[0, 4, 0] = 0.9
    raw[0, 5, 0] = 0.8
    dets = postprocess_detections(
        raw, max_det=300, nc=nc, nms=True, conf_thres=0.25, nms_iou=0.7, multi_label=True
    )
    valid = _valid_rows(dets[0])
    assert valid.shape[0] == 2
    assert set(valid[:, 5].tolist()) == {0.0, 1.0}


def test_empty_image_is_zero_padded():
    raw = torch.zeros(1, 4 + 3, 8)
    raw[:, :4] = 10.0
    dets = postprocess_detections(
        raw, max_det=10, nc=3, nms=True, conf_thres=0.25, nms_iou=0.7
    )
    assert dets.shape == (1, 10, 6)
    assert torch.count_nonzero(dets) == 0


def test_nms_keeps_separated_boxes_after_crowded_cluster():
    """先 top-300 会丢掉低分但互不重叠的框；官方 NMS 会留下它们。"""
    nc, n_crowd, n_spread, max_det = 1, 200, 200, 300
    anchors = n_crowd + n_spread
    raw = torch.zeros(1, 4 + nc, anchors)
    for i in range(n_crowd):
        raw[0, 0, i] = 100.0
        raw[0, 1, i] = 100.0
        raw[0, 2, i] = 20.0
        raw[0, 3, i] = 20.0
        raw[0, 4, i] = 0.95
    for i in range(n_spread):
        j = n_crowd + i
        raw[0, 0, j] = 40.0 + i * 25.0
        raw[0, 1, j] = 400.0
        raw[0, 2, j] = 8.0
        raw[0, 3, j] = 8.0
        raw[0, 4, j] = 0.50

    old = apply_nms_to_detections(
        Detect.postprocess(raw.permute(0, 2, 1), max_det, nc),
        conf_thres=0.25,
        nms_iou=0.7,
    )
    new = postprocess_detections(
        raw, max_det=max_det, nc=nc, nms=True, conf_thres=0.25, nms_iou=0.7
    )
    n_old = int(_valid_rows(old[0]).shape[0])
    n_new = int(_valid_rows(new[0]).shape[0])
    assert n_new > n_old
    assert n_new == 1 + n_spread


def test_nms_false_still_uses_detect_topk():
    raw = _rand_raw(batch=1, nc=4, anchors=20)
    dets = postprocess_detections(
        raw, max_det=8, nc=4, nms=False, conf_thres=0.001, nms_iou=0.7
    )
    ref = Detect.postprocess(raw.permute(0, 2, 1), 8, 4)
    assert torch.equal(dets, ref)
