# -*- coding: utf-8 -*-
"""score_classifier_dump 的映射解析和准确率。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "score_classifier_dump.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("score_classifier_dump", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def score_dump():
    return _load_script()


def test_parse_class_groups_maps_raw_to_train(score_dump):
    assert score_dump.parse_class_groups("0:0;1:2") == {0: 0, 2: 1}
    assert score_dump.parse_class_groups(" 0:0,1 ; 1:2 ") == {0: 0, 1: 0, 2: 1}


def test_parse_class_groups_rejects_bad_maps(score_dump):
    with pytest.raises(ValueError, match="duplicate train id"):
        score_dump.parse_class_groups("0:0;0:1")
    with pytest.raises(ValueError, match="more than one group"):
        score_dump.parse_class_groups("0:0;1:0")
    with pytest.raises(ValueError, match="0..1"):
        score_dump.parse_class_groups("1:2")


def test_score_pairs_skips_unmapped_and_counts_macro(score_dump):
    pairs = [(0, 0), (0, 1), (2, 1), (2, 0), (1, 0), (2, 9)]
    report = score_dump.score_pairs(pairs, {0: 0, 2: 1})
    assert report["used"] == 5
    assert report["skip_by_id"] == {1: 1}
    assert report["correct"] == 2
    assert report["pred_out_of_range"] == 1
    assert report["matrix"].tolist() == [[1, 1], [1, 1]]
    assert report["recalls"][0] == 1 / 2
    assert report["recalls"][1] == 1 / 3
    assert report["acc_macro"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_load_pairs_reads_class_id_and_pred_id(score_dump, tmp_path: Path):
    path = tmp_path / "dump.json"
    path.write_text(
        json.dumps(
            {
                "summary": {"acc1": 0.0},
                "samples": [{"class_id": 2, "pred_id": 1}],
            }
        ),
        encoding="utf-8",
    )
    assert score_dump.load_pairs(path) == [(2, 1)]
