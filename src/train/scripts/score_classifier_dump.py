#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 class_groups 把分类 dump 的原始 class_id 映成训练 id，再算准确率。

板端 ``classifier_dump_sigmastar`` 和
``exports/sigmastar/mobilenet_v2/verify/multi_image_accuracy_dump_onnx_json.py``
都把 CSV 的原始 ``class_id`` 和模型 ``pred_id`` 写进 JSON。两边的
``meta.acc1`` / ``summary.acc1`` 是这两个数直接比较，没有做训练标签映射。
裁框实验的 YAML 通常把若干原始 id 合成训练 id，这份脚本做的就是那一步。

在 ``train`` 仓库根目录执行::

    python scripts/score_classifier_dump.py \\
        --dump tmp/v013_gray.json \\
        --dump tmp/v013_onnx.json \\
        --class-groups '0:0;1:2'

参数
----

``--dump``
    分类 dump JSON，可重复。同一条映射依次打在每份文件上，方便把板端和 ONNX
    放在一次命令里。只读 ``samples`` 里的 ``class_id`` 和 ``pred_id``。
    不读 ``logits``，也不采用 ``meta`` / ``summary`` 里已经写好的 acc。
    样本缺这两列、JSON 没有 ``samples``、或文件打不开时，该份记为失败。

``--class-groups``
    与实验 YAML 的 ``class_groups`` 同向：训练 id 在冒号左边，原始 ``class_id``
    在右边。组与组用分号分开，一组里多个原始 id 用逗号::

        0:0;1:2          对应 YAML 的 0: [0]、1: [2]
        0:0,1;1:2       训练 0 合并原始 0 和 1，训练 1 来自原始 2

    训练 id 必须正好是 ``0 .. n-1``，不能跳号，也不能重复。原始 id 不能出现在
    两组里。空串、缺冒号、非整数都会退出，退出码为 1。

哪些样本进入指标
----------------

原始 ``class_id`` 不在任何组里的样本计入 ``skip``，并按原始 id 分行打印条数。
v013 的 ``1`` 不在 ``0: [0]、1: [2]`` 里，所以这 67 条不进准确率。

留下的样本用映射后的训练 id 当真值，和 ``pred_id`` 比较。``pred_id`` 不在
``0 .. n-1`` 时这条算错：recall 的分母仍算它，混淆矩阵没有这一列，另外打印
``pred out of range``。2 类模型的 argmax 不会走出这个范围。

映射之后一个训练类都没有样本时，该份退出码为 1。多份 dump 里只要有一份失败，
进程退出码就是 1；已经打出来的报告仍然有效。

打印什么
--------

``samples``
    映射后留下的条数，括号里是 ``skip`` 总数。
``confusion matrix``
    行是映射后的真值，列是 ``pred_id``，下标都是训练 id。
``acc``
    全体准确率，对上的条数 / 留下的条数。对应训练侧
    ``ImageClassifierMetric(average="micro")`` 的 ``acc``。
``acc_macro``
    各类 recall 的平均。某个训练 id 没有真值时该类不参与平均，并在 recall 里
    打印 ``n/a``。对应 ``average="macro"`` 的 ``acc_macro``，也是框级实验
    checkpoint 监控的 ``val_acc_macro``。
``per-class recall``
    该类对上的条数 / 该类留下的条数。分母含 ``pred_id`` 越界的样本。
``per-class precision``
    混淆矩阵对角线 / 该列之和。列和只统计 ``pred_id`` 落在 ``0 .. n-1`` 的样本。
    该列没有预测时打印 ``n/a``。

不计算 top-5。那需要 ``logits``，框级 2 类实验也不看它。整图 ImageNet 的
``class_id`` 已经是模型类别下标，直接读 JSON 里的 ``acc1`` / ``acc5`` 即可，不必跑这份脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score classifier dump JSON after class_groups mapping")
    p.add_argument(
        "--dump",
        action="append",
        required=True,
        help="Dump JSON. Repeat to score several files with the same map",
    )
    p.add_argument(
        "--class-groups",
        required=True,
        help="train_id:raw[,raw];... same direction as YAML class_groups. Example: 0:0;1:2",
    )
    return p.parse_args()


def parse_class_groups(text: str) -> dict[int, int]:
    """返回 ``{原始 class_id: 训练 id}``。"""
    body = text.strip()
    if not body:
        raise ValueError("--class-groups is empty")
    raw_to_train: dict[int, int] = {}
    train_ids: set[int] = set()
    for part in body.split(";"):
        item = part.strip()
        if item == "" or ":" not in item:
            raise ValueError(f"class group {part!r} must look like train_id:raw[,raw]")
        left, right = item.split(":", 1)
        try:
            train_id = int(left.strip())
        except ValueError:
            raise ValueError(f"train id {left!r} is not an int") from None
        if train_id in train_ids:
            raise ValueError(f"duplicate train id {train_id}")
        raws: list[int] = []
        for piece in right.split(","):
            token = piece.strip()
            if token == "":
                raise ValueError(f"class_groups {train_id} has an empty raw id")
            try:
                raw = int(token)
            except ValueError:
                raise ValueError(f"raw class_id {token!r} is not an int") from None
            if raw in raw_to_train:
                raise ValueError(f"raw class_id {raw} is in more than one group")
            raws.append(raw)
        if not raws:
            raise ValueError(f"class_groups {train_id} is empty")
        train_ids.add(train_id)
        for raw in raws:
            raw_to_train[raw] = train_id
    last = max(train_ids)
    if train_ids != set(range(last + 1)):
        raise ValueError(f"train ids must be 0..{last}, got {sorted(train_ids)}")
    return raw_to_train


def load_pairs(dump_path: Path) -> list[tuple[int, int]] | None:
    try:
        data = json.loads(dump_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid json: {exc}", file=sys.stderr)
        return None
    samples = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, list):
        print("json missing samples array", file=sys.stderr)
        return None
    pairs: list[tuple[int, int]] = []
    for index, row in enumerate(samples):
        if not isinstance(row, dict) or "class_id" not in row or "pred_id" not in row:
            print(f"sample {index} missing class_id or pred_id", file=sys.stderr)
            return None
        pairs.append((int(row["class_id"]), int(row["pred_id"])))
    return pairs


def score_pairs(
    pairs: list[tuple[int, int]],
    raw_to_train: dict[int, int],
) -> dict[str, Any]:
    num_classes = max(raw_to_train.values()) + 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    gt_total = np.zeros(num_classes, dtype=np.int64)
    gt_hit = np.zeros(num_classes, dtype=np.int64)
    skip_by_id: dict[int, int] = {}
    pred_out_of_range = 0
    for class_id, pred_id in pairs:
        if class_id not in raw_to_train:
            skip_by_id[class_id] = skip_by_id.get(class_id, 0) + 1
            continue
        label = raw_to_train[class_id]
        gt_total[label] += 1
        if pred_id == label:
            gt_hit[label] += 1
        if 0 <= pred_id < num_classes:
            matrix[label, pred_id] += 1
        else:
            pred_out_of_range += 1
    used = int(gt_total.sum())
    correct = int(gt_hit.sum())
    recalls: list[float | None] = []
    for index in range(num_classes):
        total = int(gt_total[index])
        recalls.append(None if total == 0 else int(gt_hit[index]) / total)
    present = [value for value in recalls if value is not None]
    acc_macro = None if not present else float(sum(present) / len(present))
    return {
        "num_classes": num_classes,
        "matrix": matrix,
        "gt_total": gt_total,
        "gt_hit": gt_hit,
        "skip_by_id": skip_by_id,
        "pred_out_of_range": pred_out_of_range,
        "used": used,
        "correct": correct,
        "acc": None if used == 0 else correct / used,
        "recalls": recalls,
        "acc_macro": acc_macro,
    }


def print_report(dump_path: Path, report: dict[str, Any]) -> None:
    num_classes = int(report["num_classes"])
    matrix: np.ndarray = report["matrix"]
    gt_total: np.ndarray = report["gt_total"]
    gt_hit: np.ndarray = report["gt_hit"]
    skip_by_id: dict[int, int] = report["skip_by_id"]
    skip = sum(skip_by_id.values())
    used = int(report["used"])
    correct = int(report["correct"])
    print(f"dump: {dump_path}")
    print(f"samples: {used} (skip: {skip})")
    for class_id in sorted(skip_by_id):
        print(f"skip class_id {class_id}: {skip_by_id[class_id]}")
    if report["pred_out_of_range"]:
        print(f"pred out of range: {report['pred_out_of_range']}")
    print()
    print("confusion matrix (rows=gt, cols=pred):")
    header = "gt\\pred".ljust(10) + "".join(f"{index:>8d}" for index in range(num_classes))
    print(header)
    for row in range(num_classes):
        print(f"{row:<10d}" + "".join(f"{int(matrix[row, col]):8d}" for col in range(num_classes)))
    print()
    acc = report["acc"]
    acc_macro = report["acc_macro"]
    print(f"acc: {correct}/{used} = {acc:.6f}")
    if acc_macro is None:
        print("acc_macro: n/a")
    else:
        print(f"acc_macro: {acc_macro:.6f}")
    print("per-class recall:")
    for index, recall in enumerate(report["recalls"]):
        total = int(gt_total[index])
        if recall is None:
            print(f"  class {index}: n/a (no GT)")
        else:
            print(f"  class {index}: {int(gt_hit[index])}/{total} = {recall:.6f}")
    print("per-class precision:")
    for col in range(num_classes):
        col_sum = int(matrix[:, col].sum())
        hit = int(matrix[col, col])
        if col_sum == 0:
            print(f"  class {col}: n/a (no pred)")
        else:
            print(f"  class {col}: {hit}/{col_sum} = {hit / col_sum:.6f}")


def score_dump(dump_path: Path, raw_to_train: dict[int, int]) -> int:
    if not dump_path.is_file():
        print(f"dump not found: {dump_path}", file=sys.stderr)
        return 1
    pairs = load_pairs(dump_path)
    if pairs is None:
        return 1
    report = score_pairs(pairs, raw_to_train)
    if report["used"] == 0:
        print(f"no samples left after class_groups: {dump_path}", file=sys.stderr)
        return 1
    print_report(dump_path, report)
    return 0


def main() -> int:
    args = parse_args()
    try:
        raw_to_train = parse_class_groups(args.class_groups)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    status = 0
    for index, dump in enumerate(args.dump):
        if index:
            print()
        status |= score_dump(Path(dump), raw_to_train)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
