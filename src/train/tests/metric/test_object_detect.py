from __future__ import annotations

import torch

from lovely_deep_learning.metric.object_detect import ObjectDetectMetric


def _pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _gt(boxes, labels):
    return {
        "bboxes_xyxy_abs_tv_transformed": torch.tensor(boxes, dtype=torch.float32),
        "cls_tv_transformed": torch.tensor(labels, dtype=torch.long),
    }


def test_perfect_match_map_is_one():
    m = ObjectDetectMetric(box_format="xyxy", nc=2)
    preds = [_pred([[10.0, 10.0, 30.0, 30.0]], [0.9], [0])]
    net_out = (_gt([[10.0, 10.0, 30.0, 30.0]], [0]),)
    m.update("val", preds, net_out)
    out = m.compute("val")
    assert float(out["map"]) >= 0.99
    assert float(out["map_50"]) >= 0.99


def test_compute_is_not_negative_one():
    m = ObjectDetectMetric(box_format="xyxy", nc=80)
    preds = [_pred([[0.0, 0.0, 10.0, 10.0]], [0.5], [1])]
    net_out = (_gt([[1.0, 1.0, 11.0, 11.0]], [1]),)
    m.update("val", preds, net_out)
    out = m.compute("val")
    assert float(out["map"]) >= 0.0
    assert float(out["map"]) != -1.0
