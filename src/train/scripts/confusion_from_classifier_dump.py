#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从分类 dump CSV（板端或 ONNX）计算 3×3 混淆矩阵与准确率。

Dump 列需含 ``gt_id,pred_id``（与板端 ``image_classifiter_from_detect_box_sigmastar`` /
``eval_box_crop_classifier_onnx.py`` 一致）。

示例::

  python scripts/confusion_from_classifier_dump.py --dump dump_board.csv
  python scripts/confusion_from_classifier_dump.py --dump dump_onnx.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confusion matrix from classifier dump CSV")
    p.add_argument("--dump", required=True, help="dump CSV with gt_id,pred_id")
    p.add_argument("--num-classes", type=int, default=3, help="matrix size (default: 3)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dump_path = Path(args.dump)
    if not dump_path.is_file():
        print(f"dump not found: {dump_path}", file=sys.stderr)
        return 1
    if args.num_classes <= 0:
        print(f"invalid --num-classes: {args.num_classes}", file=sys.stderr)
        return 1

    n = args.num_classes
    cm = np.zeros((n, n), dtype=np.int64)
    total = 0
    skipped = 0

    with dump_path.open(newline="") as ifs:
        reader = csv.DictReader(ifs)
        if reader.fieldnames is None or not {"gt_id", "pred_id"}.issubset(set(reader.fieldnames)):
            print("dump must have gt_id,pred_id columns", file=sys.stderr)
            return 1
        for row in reader:
            gt = int(float(row["gt_id"]))
            pred = int(float(row["pred_id"]))
            if gt < 0 or gt >= n or pred < 0 or pred >= n:
                skipped += 1
                continue
            cm[gt, pred] += 1
            total += 1

    if total == 0:
        print("no valid rows", file=sys.stderr)
        return 1

    print(f"dump: {dump_path}")
    print(f"samples: {total} (skipped out-of-range: {skipped})")
    print()
    print("confusion matrix (rows=gt, cols=pred):")
    header = "gt\\pred".ljust(10) + "".join(f"{j:>8d}" for j in range(n))
    print(header)
    for i in range(n):
        print(f"{i:<10d}" + "".join(f"{cm[i, j]:8d}" for j in range(n)))

    correct = int(np.trace(cm))
    acc = correct / total
    print()
    print(f"overall accuracy: {correct}/{total} = {acc:.6f}")

    print("per-class recall (diag / row sum):")
    for i in range(n):
        row_sum = int(cm[i].sum())
        if row_sum == 0:
            print(f"  class {i}: n/a (no GT)")
        else:
            print(f"  class {i}: {cm[i, i]}/{row_sum} = {cm[i, i] / row_sum:.6f}")

    print("per-class precision (diag / col sum):")
    for j in range(n):
        col_sum = int(cm[:, j].sum())
        if col_sum == 0:
            print(f"  class {j}: n/a (no pred)")
        else:
            print(f"  class {j}: {cm[j, j]}/{col_sum} = {cm[j, j] / col_sum:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
