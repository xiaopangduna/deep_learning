# -*- coding: utf-8 -*-
"""Unit tests for class-map evaluation in val_classifier_with_fiftyone."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "val_classifier_with_fiftyone.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("val_classifier_with_fiftyone", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vc():
    return _load_script()


def test_parse_class_map_is_one_to_one(vc):
    parsed = vc.parse_class_map("0:face_frontal,1:face_side,2:face_head")
    assert parsed.name_to_id == {
        "face_frontal": 0,
        "face_side": 1,
        "face_head": 2,
    }
    assert parsed.first_name(0) == "face_frontal"
    assert vc.parse_class_map("0:eye_open_0，1:eye_open_2").ids() == {0, 1}
    with pytest.raises(ValueError, match="id 重复"):
        vc.parse_class_map("0:eye_open_0,0:eye_open_1")
    with pytest.raises(ValueError, match="类名重复"):
        vc.parse_class_map("0:eye_open_0,1:eye_open_0")
    with pytest.raises(ValueError, match="id:类名"):
        vc.parse_class_map("face_frontal")


def test_validate_class_map_rejects_unknown_background_and_collisions(vc):
    class_map = vc.parse_class_map("0:eye_open_0,1:eye_open_2")
    groups = {0: [0], 1: [2]}
    vc.validate_class_map(class_map, groups, None, ["head"])
    with pytest.raises(ValueError, match="不在 class_groups"):
        vc.validate_class_map(class_map, {0: [0]}, None, ["head"])
    background = vc.parse_class_map("0:eye_open_0")
    with pytest.raises(ValueError, match="background"):
        vc.validate_class_map(background, {1: [0]}, 0, ["head"])
    prefixed = vc.parse_class_map("0:head-eye_open_0,1:eye_open_0")
    with pytest.raises(ValueError, match="不要带框标签前缀"):
        vc.validate_class_map(prefixed, {0: [0], 1: [1]}, None, ["head"])


def test_matched_names_are_exact_box_tags(vc):
    class_map = vc.parse_class_map("0:eye_open_0,1:eye_open_1,2:eye_open_2")
    assert vc.matched_class_names("head", ["eye_open_0", "changed"], class_map) == [
        "eye_open_0"
    ]
    assert vc.matched_class_names("head", ["changed", "eval_err"], class_map) == []
    assert vc.matched_class_names(
        "head",
        ["eye_open_0", "eye_open_1"],
        class_map,
    ) == ["eye_open_0", "eye_open_1"]
    prefixed = vc.parse_class_map("0:head-face_frontal,1:head-face_side")
    assert vc.matched_class_names("head", ["face_side"], prefixed) == []
    assert vc.matched_class_names("head", ["head-face_frontal"], prefixed) == [
        "head-face_frontal"
    ]


def test_pred_tag_uses_mapped_name_or_raw_id(vc):
    class_map = vc.parse_class_map("0:eye_open_0,1:eye_open_2")
    assert vc.pred_tag_for_id(0, class_map) == "pred_eye_open_0"
    assert vc.pred_tag_for_id(1, class_map) == "pred_eye_open_2"
    assert vc.pred_tag_for_id(2, class_map) == "pred_2"


def test_apply_eval_tags_keeps_ground_truth_tags(vc):
    tags = vc.apply_eval_tags(
        ["eye_open_0", "changed", "eval_ok", "pred_eye_open_2"],
        status=vc.STATUS_ERR,
        pred_tag="pred_eye_open_0",
    )
    assert tags == ["eye_open_0", "changed", "eval_err", "pred_eye_open_0"]
    correct = vc.apply_eval_tags(
        ["eye_open_2", "eval_ok", "pred_eye_open_0"],
        status=vc.STATUS_OK,
        pred_tag="pred_eye_open_2",
    )
    assert correct == ["eye_open_2", "pred_eye_open_2"]
    unlabeled = vc.apply_eval_tags(
        ["changed", "eval_err", "pred_eye_open_2"],
        status=vc.STATUS_BAD,
        pred_tag="pred_eye_open_0",
    )
    assert unlabeled == ["changed", "eval_bad_tag", "pred_eye_open_0"]
    multi = vc.apply_eval_tags(
        ["eye_open_0", "eye_open_2", "eval_err", "pred_eye_open_0"],
        status=vc.STATUS_BAD,
        pred_tag="",
    )
    assert multi == ["eye_open_0", "eye_open_2", "eval_bad_tag"]


def test_eval_class_groups_is_identity_over_model_ids(vc):
    synthetic = vc.eval_class_groups({0: [0, 1], 1: [2]}, 2)
    parsed, background_id = vc.ImageClassifierDataset.parse_class_groups(synthetic)
    assert background_id == 2
    assert parsed == {0: [0], 1: [1]}


def test_bad_tags_stay_out_of_error_rate(vc):
    stats = vc.EvalStats()
    stats.add(
        vc.BoxVerdict(
            sample_id="s",
            detection_id="d1",
            img_path="a.jpg",
            img_w=10,
            img_h=10,
            cx=0.5,
            cy=0.5,
            w=0.2,
            h=0.2,
            status=vc.STATUS_OK,
            gt_name="head-face_frontal",
            gt_id=0,
        )
    )
    stats.add(
        vc.BoxVerdict(
            sample_id="s",
            detection_id="d2",
            img_path="a.jpg",
            img_w=10,
            img_h=10,
            cx=0.5,
            cy=0.5,
            w=0.2,
            h=0.2,
            status=vc.STATUS_BAD,
            bad_reason=vc.REASON_MULTI_GT,
            matched_names=("head-eye_open_0", "head-eye_open_1"),
        )
    )
    assert stats.n_scored == 1
    assert stats.n_ok == 1
    assert stats.n_err == 0
    assert stats.n_bad == 1
    assert stats.n_multi_gt == 1
