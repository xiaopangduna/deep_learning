# -*- coding: utf-8 -*-
"""眼睛预标注的映射、几何筛选和写标签规则。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prelabel_eye_fiftyone.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prelabel_eye_fiftyone", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prelabel():
    return _load_script()


def test_parse_tag_map_accepts_ascii_and_fullwidth_comma(prelabel):
    assert prelabel.parse_tag_map("1:eye_open_0,2:eye_open_2") == {1: "eye_open_0", 2: "eye_open_2"}
    assert prelabel.parse_tag_map("1:eye_open_0，2:eye_open_2") == {1: "eye_open_0", 2: "eye_open_2"}


def test_parse_tag_map_rejects_bad_entries(prelabel):
    with pytest.raises(ValueError, match="eye_open_"):
        prelabel.parse_tag_map("1:closed")
    with pytest.raises(ValueError, match="重复"):
        prelabel.parse_tag_map("1:eye_open_0,1:eye_open_2")
    with pytest.raises(ValueError, match="不能为空"):
        prelabel.parse_tag_map(" , ")


def test_validate_tag_map_allows_omitting_background(prelabel):
    groups = {1: [0, 1], 2: [2]}
    tag_map = prelabel.parse_tag_map("1:eye_open_0,2:eye_open_2")
    prelabel.validate_tag_map(tag_map, groups, 0)
    with pytest.raises(ValueError, match="不在 class_groups"):
        prelabel.validate_tag_map({9: "eye_open_0"}, groups, 0)


def test_decide_closed_open_and_ignored_background(prelabel):
    tag_map = {1: "eye_open_0", 2: "eye_open_2"}
    assert prelabel.decide_label(
        pred_group_id=1,
        pred_conf=0.9,
        tag_map=tag_map,
        existing=[],
        min_conf=0.8,
        overwrite=False,
    ) == ("write", "eye_open_0")
    assert prelabel.decide_label(
        pred_group_id=2,
        pred_conf=0.91,
        tag_map=tag_map,
        existing=[],
        min_conf=0.8,
        overwrite=False,
    ) == ("write", "eye_open_2")
    assert prelabel.decide_label(
        pred_group_id=0,
        pred_conf=0.99,
        tag_map=tag_map,
        existing=[],
        min_conf=0.8,
        overwrite=False,
    ) == ("unmapped", None)


def test_decide_skips_existing_unless_overwrite_and_keeps_low_conf(prelabel):
    tag_map = {1: "eye_open_0", 2: "eye_open_2"}
    assert prelabel.decide_label(
        pred_group_id=2,
        pred_conf=0.99,
        tag_map=tag_map,
        existing=["eye_open_1"],
        min_conf=0.8,
        overwrite=False,
    ) == ("skip_existing", None)
    assert prelabel.decide_label(
        pred_group_id=2,
        pred_conf=0.99,
        tag_map=tag_map,
        existing=["eye_open_1", "ignore"],
        min_conf=0.8,
        overwrite=True,
    ) == ("write", "eye_open_2")
    assert prelabel.decide_label(
        pred_group_id=2,
        pred_conf=0.5,
        tag_map=tag_map,
        existing=[],
        min_conf=0.8,
        overwrite=False,
    ) == ("low_conf", None)
    assert prelabel.decide_label(
        pred_group_id=0,
        pred_conf=0.2,
        tag_map=tag_map,
        existing=["eye_open_0"],
        min_conf=0.8,
        overwrite=True,
    ) == ("unmapped", None)
    assert prelabel.decide_label(
        pred_group_id=2,
        pred_conf=0.99,
        tag_map=tag_map,
        existing=["eye_open_2", "changed"],
        min_conf=0.8,
        overwrite=True,
    ) == ("same", "eye_open_2")


def test_replace_eye_tag_keeps_other_tags(prelabel):
    assert prelabel.replace_eye_tag(["ignore", "eye_open_1", "eye_open_0"], "eye_open_2") == [
        "ignore",
        "eye_open_2",
    ]


def test_changed_tag_marks_boxes_whose_eye_tag_changes(prelabel):
    assert prelabel.tags_after_write(["ignore", "eye_open_1"], "eye_open_2") == [
        "ignore",
        "eye_open_2",
        "changed",
    ]
    assert prelabel.tags_after_write(["ignore", "changed", "eye_open_1"], "eye_open_0") == [
        "ignore",
        "changed",
        "eye_open_0",
    ]
    assert prelabel.tags_after_write([], "eye_open_0") == ["eye_open_0", "changed"]


def test_box_conversion_and_short_side(prelabel):
    assert prelabel.fo_xywh_to_yolo_cxcywh([0.4, 0.3, 0.2, 0.4]) == pytest.approx((0.5, 0.5, 0.2, 0.4))
    assert prelabel.geometry_status(0.01, 0.01, 100, 100, 16) == "short_side"
    assert prelabel.geometry_status(0.5, 0.5, 100, 100, 16) is None
    assert prelabel.geometry_status(0.0, 0.2, 100, 100, 16) == "invalid_box"
