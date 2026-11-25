import torch
import torch.nn.functional as F

from lovely_deep_learning.loss.attribute_classifier import AttributeClassifierCriterion
from lovely_deep_learning.metric.attribute_classifier import AttributeClassifierMetric
from lovely_deep_learning.postprocess.attribute_classifier import AttributeClassifierPostProcessor


def test_bce_matches_functional_and_pos_weight_changes_it():
    logits = torch.tensor([[2.0, -2.0], [0.0, 3.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    criterion = AttributeClassifierCriterion()
    loss = criterion([logits], net_out={"attr": target})
    assert torch.isclose(loss, F.binary_cross_entropy_with_logits(logits, target))

    weighted = AttributeClassifierCriterion(pos_weight=[1.0, 4.0])
    weighted_loss = weighted([logits], net_out={"attr": target})
    expected = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor([1.0, 4.0])
    )
    assert torch.isclose(weighted_loss, expected)
    assert not torch.isclose(weighted_loss, loss)


def test_postprocess_thresholds_sigmoid():
    logits = torch.tensor([[2.0, -2.0]])
    out = AttributeClassifierPostProcessor(threshold=0.5).run([logits])
    assert torch.allclose(out["probs"], torch.sigmoid(logits))
    assert torch.equal(out["pred"], torch.tensor([[1.0, 0.0]]))


def test_mean_accuracy_weights_attributes_equally():
    metric = AttributeClassifierMetric(num_attributes=2)
    pred = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    metric.update("val", {"pred": pred}, {"attr": target})
    score = metric.compute("val")
    assert set(score) == {"acc"}
    assert float(score["acc"]) == 0.5
    metric.reset("val")
    assert float(metric.compute("val")["acc"]) == 0.0
