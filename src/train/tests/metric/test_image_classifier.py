from __future__ import annotations

import pytest
import torch

from lovely_deep_learning.metric.image_classifier import ImageClassifierMetric


def _batch(targets: list[int], pred_ids: list[int]) -> tuple[dict, dict]:
    logits = torch.full((len(targets), 2), -8.0)
    for index, pred in enumerate(pred_ids):
        logits[index, pred] = 8.0
    return {"logits": logits}, {"class_id": torch.tensor(targets)}


def test_micro_accuracy_is_overall_correct_rate():
    metric = ImageClassifierMetric(num_classes=2)
    post_out, net_out = _batch([0] * 845 + [1] * 155, [0] * 1000)
    metric.update("val", post_out, net_out)
    score = metric.compute("val")
    assert set(score) == {"acc"}
    assert float(score["acc"]) == pytest.approx(0.845)


def test_macro_accuracy_is_mean_of_class_recalls():
    metric = ImageClassifierMetric(num_classes=2, average="macro")
    targets = [0] * 845 + [1] * 155
    preds = [0] * 805 + [1] * 40 + [1] * 93 + [0] * 62
    metric.update("val", *_batch(targets, preds))
    score = metric.compute("val")
    assert set(score) == {"acc_macro"}
    assert float(score["acc_macro"]) == pytest.approx((805 / 845 + 93 / 155) / 2)


def test_macro_accuracy_of_majority_guess_is_one_half():
    metric = ImageClassifierMetric(num_classes=2, average="macro")
    metric.update("val", *_batch([0] * 845 + [1] * 155, [0] * 1000))
    assert float(metric.compute("val")["acc_macro"]) == pytest.approx(0.5)


def test_average_rejects_weighted():
    with pytest.raises(ValueError, match="micro 或 macro"):
        ImageClassifierMetric(num_classes=2, average="weighted")
