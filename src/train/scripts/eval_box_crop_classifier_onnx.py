#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原图 + CSV → ONNX 分类 dump（列格式与板端 dump 一致）。

预处理对齐 ``ImageClassifierFromDetectBoxDataset`` / 训练 transform_val：
  BGR 读图 → crop_cxcywh(box_scale) → LetterBox(fill=114) → RGB /255 → ImageNet Normalize

默认标签映射 A（与 ``image_classifiter_from_detect_box_head_eye_open`` 的 class_groups 一致）：
  CSV class_id 0/1 → gt_id=1，2 → gt_id=2。不合成 background。

板端只 dump；精度与混淆矩阵在训练侧用本脚本 + ``confusion_from_classifier_dump.py``。

示例::

  python scripts/eval_box_crop_classifier_onnx.py \\
      --model path/to/eye_v007.onnx \\
      --csv path/to/v301_eye_copy/test_v301.csv \\
      --out dump_onnx.csv \\
      --box-scale 1.2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from lovely_deep_learning.dataset.base import BaseDataset
from lovely_deep_learning.dataset.image_classifier import ImageClassifierFromDetectBoxDataset
from lovely_deep_learning.transforms.letterbox import letterbox_resize_and_pad

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 onnxruntime：pip install onnxruntime") from exc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LETTERBOX_FILL = 114
DEFAULT_OUT = "dump_onnx.csv"
DEFAULT_BOX_SCALE = 1.2
NUM_CLASSES = 3

# 标签映射 A：与 head_eye_open 实验 class_groups 对齐
DEFAULT_GT_MAP: Dict[int, int] = {0: 1, 1: 1, 2: 2}


def map_gt_id(class_id: int, gt_map: Dict[int, int]) -> Optional[int]:
    return gt_map.get(class_id)


def letterbox_bgr(src: np.ndarray, dst_w: int, dst_h: int, fill: int = LETTERBOX_FILL) -> np.ndarray:
    """用训练侧 ``letterbox_resize_and_pad`` 几何，OpenCV 执行 resize/pad。"""
    src_h, src_w = src.shape[:2]
    resize_h, resize_w, left, top, right, bottom = letterbox_resize_and_pad(
        src_h, src_w, dst_h, dst_w, scaleup=True, center=True
    )
    resized = cv2.resize(src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(fill, fill, fill),
    )
    if padded.shape[0] != dst_h or padded.shape[1] != dst_w:
        exact = np.full((dst_h, dst_w, src.shape[2]), fill, dtype=src.dtype)
        copy_h = min(padded.shape[0], dst_h)
        copy_w = min(padded.shape[1], dst_w)
        exact[:copy_h, :copy_w] = padded[:copy_h, :copy_w]
        return exact
    return padded


def preprocess_bgr(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    box_scale: float,
    model_w: int,
    model_h: int,
) -> np.ndarray:
    crop = ImageClassifierFromDetectBoxDataset.crop_cxcywh(bgr, cx, cy, w, h, box_scale)
    letterboxed = letterbox_bgr(crop, model_w, model_h, LETTERBOX_FILL)
    # 与 Dataset：BGR → RGB CHW uint8 → float /255 → Normalize
    img_tv = BaseDataset.convert_img_from_numpy_to_tensor_uint8(letterboxed)
    img_tv = F.to_dtype(img_tv, dtype=torch.float32, scale=True)
    img_tv = F.normalize(img_tv, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    return img_tv.unsqueeze(0).contiguous().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ONNX box-crop classifier dump (same CSV schema as board dump)"
    )
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--csv", required=True, help="annotation CSV; path_img relative to CSV dir")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"dump CSV (default: {DEFAULT_OUT})")
    p.add_argument(
        "--box-scale",
        type=float,
        default=DEFAULT_BOX_SCALE,
        help=f"crop expand factor (default: {DEFAULT_BOX_SCALE})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.box_scale <= 0:
        print(f"invalid --box-scale: {args.box_scale}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    csv_path = Path(args.csv).resolve()
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not csv_path.is_file():
        print(f"csv not found: {csv_path}", file=sys.stderr)
        return 1

    csv_dir = csv_path.parent
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape
    model_h = int(shape[2]) if isinstance(shape[2], int) else 224
    model_w = int(shape[3]) if isinstance(shape[3], int) else 224

    print("eval_box_crop_classifier_onnx.py")
    print(f"  model     : {model_path}")
    print(f"  csv       : {csv_path}")
    print(f"  out       : {args.out}")
    print(f"  box-scale : {args.box_scale}")
    print(f"  input     : {input_name} {[1, 3, model_h, model_w]}")

    ok_count = 0
    skip_count = 0
    with csv_path.open(newline="") as ifs, Path(args.out).open("w", newline="") as ofs:
        reader = csv.DictReader(ifs)
        required = {"path_img", "class_id", "cx", "cy", "w", "h"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            print(f"csv missing columns {required}: {csv_path}", file=sys.stderr)
            return 1
        writer = csv.writer(ofs)
        writer.writerow(
            ["path_img", "class_id", "gt_id", "logit_0", "logit_1", "logit_2", "pred_id"]
        )

        rows = list(reader)
        print(f"  samples   : {len(rows)}")
        for i, row in enumerate(rows):
            class_id = int(float(row["class_id"]))
            gt_id = map_gt_id(class_id, DEFAULT_GT_MAP)
            if gt_id is None:
                print(
                    f"[{i}] skip unmapped class_id={class_id} path={row['path_img']}",
                    file=sys.stderr,
                )
                skip_count += 1
                continue

            rel = row["path_img"]
            img_path = Path(rel) if Path(rel).is_absolute() else (csv_dir / rel)
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[{i}] failed to read: {img_path}", file=sys.stderr)
                skip_count += 1
                continue

            tensor = preprocess_bgr(
                bgr,
                float(row["cx"]),
                float(row["cy"]),
                float(row["w"]),
                float(row["h"]),
                args.box_scale,
                model_w,
                model_h,
            )
            outputs = session.run(None, {input_name: tensor})
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if logits.size < NUM_CLASSES:
                print(f"[{i}] output size {logits.size} < {NUM_CLASSES}", file=sys.stderr)
                skip_count += 1
                continue
            pred_id = int(np.argmax(logits[:NUM_CLASSES]))
            writer.writerow(
                [
                    rel,
                    class_id,
                    gt_id,
                    float(logits[0]),
                    float(logits[1]),
                    float(logits[2]),
                    pred_id,
                ]
            )
            ok_count += 1
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"  progress: {i + 1} / {len(rows)} (ok={ok_count} skip={skip_count})")

    print(f"done: wrote {ok_count} rows to {args.out} (skipped {skip_count})")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
