# -*- coding: utf-8 -*-
"""Unit tests for care-classes ERR rules in val_classifier_with_fiftyone."""

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


def test_resolve_care_by_id_name_and_suffix(vc):
    id_to_name = {
        0: "head-eye_open_0",
        1: "head-eye_open_1",
        2: "head-eye_open_2",
    }
    assert vc.resolve_care_classes("2", id_to_name) == {2}
    assert vc.resolve_care_classes("head-eye_open_2", id_to_name) == {2}
    assert vc.resolve_care_classes("eye_open_2", id_to_name) == {2}
    assert vc.resolve_care_classes("0,2", id_to_name) == {0, 2}
    assert vc.resolve_care_classes(None, id_to_name) is None
    with pytest.raises(ValueError, match="unknown"):
        vc.resolve_care_classes("eye_open_9", id_to_name)


def test_care_xor_ignores_0_vs_1(vc):
    groups = {0: [0, 1], 1: [2]}
    original_to_group = {0: 0, 1: 0, 2: 1}
    care = {2}

    # 0↔1 via pred group 0: both not in care → OK
    assert not vc.is_error(
        gt_original_id=0,
        pred_group_id=0,
        care_ids=care,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )
    assert not vc.is_error(
        gt_original_id=1,
        pred_group_id=0,
        care_ids=care,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )

    # gt=2 pred=group0 → ERR
    assert vc.is_error(
        gt_original_id=2,
        pred_group_id=0,
        care_ids=care,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )
    # gt=0 pred=group1 (open) → ERR
    assert vc.is_error(
        gt_original_id=0,
        pred_group_id=1,
        care_ids=care,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )
    # gt=2 pred=group1 → OK
    assert not vc.is_error(
        gt_original_id=2,
        pred_group_id=1,
        care_ids=care,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )


def test_exact_group_match_without_care(vc):
    groups = {0: [0, 1], 1: [2]}
    original_to_group = {0: 0, 1: 0, 2: 1}

    # 0 and 1 share group 0 — pred group 0 is OK for both
    assert not vc.is_error(
        gt_original_id=0,
        pred_group_id=0,
        care_ids=None,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )
    assert not vc.is_error(
        gt_original_id=1,
        pred_group_id=0,
        care_ids=None,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )
    assert vc.is_error(
        gt_original_id=2,
        pred_group_id=0,
        care_ids=None,
        groups=groups,
        background_id=None,
        original_to_group=original_to_group,
    )


def test_pred_tag_for_closed_open_and_background(vc):
    groups = {1: [0, 1], 2: [2]}
    assert vc.pred_tag_for_group(0, groups, 0) == "pred_background"
    assert vc.pred_tag_for_group(1, groups, 0) == "pred_eye_open_0"
    assert vc.pred_tag_for_group(2, groups, 0) == "pred_eye_open_2"


def test_apply_eval_tags_keeps_ground_truth_tags(vc):
    tags = vc.apply_eval_tags(
        ["eye_open_0", "changed", "eval_ok", "pred_eye_open_2"],
        is_err=True,
        pred_tag="pred_eye_open_0",
    )
    assert tags == ["eye_open_0", "changed", "eval_err", "pred_eye_open_0"]
    correct = vc.apply_eval_tags(
        ["eye_open_2", "eval_ok", "pred_eye_open_0"],
        is_err=False,
        pred_tag="pred_eye_open_2",
    )
    assert correct == ["eye_open_2", "pred_eye_open_2"]
    assert vc.strip_eval_tags(["eye_open_1", "eval_err", "pred_background", "ignore"]) == [
        "eye_open_1",
        "ignore",
    ]


def test_eye_open_code_uses_highest_and_defaults_to_zero(vc):
    assert vc.eye_open_code(None) == 0
    assert vc.eye_open_code(["ignore"]) == 0
    assert vc.eye_open_code(["eye_open_1", "eye_open_2"]) == 2
    assert vc.eye_class_name("head", 0) == "head-eye_open_0"
