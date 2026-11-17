#!/usr/bin/env python3
"""用 Imagenette 的 10 类映射，从板端或 ONNX 的分类 JSON 算 top-1 / top-5。

JSON 来自板端 classifier_dump_sigmastar，或本目录
multi_image_accuracy_dump_onnx_json.py。两边都读 samples 里的 class_id 和 logits。

meta.acc1 / acc5 是未映射的 class_id 对 pred_id。本脚本另外把 Imagenette
class_id 映到 ImageNet-1k 下标再算一遍。类别顺序是文件夹名字典序。

  class_id  wnid        ImageNet
  0         n01440764   0
  1         n02102040   217
  2         n02979186   482
  3         n03000684   491
  4         n03028079   497
  5         n03394916   566
  6         n03417042   569
  7         n03425413   571
  8         n03445777   574
  9         n03888257   701
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (imagenette class_id, wnid, imagenet index), folder-name order
IMAGENETTE = (
    (0, "n01440764", 0),
    (1, "n02102040", 217),
    (2, "n02979186", 482),
    (3, "n03000684", 491),
    (4, "n03028079", 497),
    (5, "n03394916", 566),
    (6, "n03417042", 569),
    (7, "n03425413", 571),
    (8, "n03445777", 574),
    (9, "n03888257", 701),
)
ID_TO_IMNET = {class_id: imagenet for class_id, _wnid, imagenet in IMAGENETTE}
ID_TO_WNID = {class_id: wnid for class_id, wnid, _imagenet in IMAGENETTE}
WNIDS = set(ID_TO_WNID.values())
TOP_K = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a classifier dump JSON with the Imagenette id map")
    p.add_argument("--dump", required=True, help="Board or ONNX JSON with samples[].logits")
    return p.parse_args()


def rank_of(logits: list[float], index: int) -> int | None:
    """0-based rank. A higher logit wins; equal logits keep the smaller index."""
    if index < 0 or index >= len(logits):
        return None
    target = logits[index]
    better = 0
    for other, logit in enumerate(logits):
        if other == index:
            continue
        if logit > target or (logit == target and other < index):
            better += 1
    return better


def wnid_in_path(path_img: str) -> str | None:
    for part in Path(path_img).parts:
        if part in WNIDS:
            return part
    return None


def load_samples(dump_path: Path) -> tuple[dict, list[dict]]:
    try:
        data = json.loads(dump_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid json: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise SystemExit("dump must be an object with a samples list")
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return meta, data["samples"]


def main() -> int:
    args = parse_args()
    dump_path = Path(args.dump)
    if not dump_path.is_file():
        print(f"dump not found: {dump_path}", file=sys.stderr)
        return 1
    meta, samples = load_samples(dump_path)
    if not samples:
        print("no samples", file=sys.stderr)
        return 1

    raw1 = raw5 = mapped1 = mapped5 = 0
    unknown = 0
    wnid_mismatch = 0
    pred_mismatch = 0
    per_n = {class_id: 0 for class_id in ID_TO_IMNET}
    per_ok = {class_id: 0 for class_id in ID_TO_IMNET}

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            print(f"sample {index} is not an object", file=sys.stderr)
            return 1
        if "class_id" not in sample or "logits" not in sample:
            print(f"sample {index} missing class_id or logits", file=sys.stderr)
            return 1
        class_id = int(sample["class_id"])
        logits = sample["logits"]
        if not isinstance(logits, list) or not logits:
            print(f"sample {index} logits are empty", file=sys.stderr)
            return 1
        raw_rank = rank_of(logits, class_id)
        if raw_rank is not None and raw_rank == 0:
            raw1 += 1
        if raw_rank is not None and raw_rank < TOP_K:
            raw5 += 1
        if "pred_id" in sample and int(sample["pred_id"]) != argmax_index(logits):
            pred_mismatch += 1

        imagenet_id = ID_TO_IMNET.get(class_id)
        if imagenet_id is None:
            unknown += 1
        else:
            per_n[class_id] += 1
            mapped_rank = rank_of(logits, imagenet_id)
            if mapped_rank is not None and mapped_rank == 0:
                mapped1 += 1
                per_ok[class_id] += 1
            if mapped_rank is not None and mapped_rank < TOP_K:
                mapped5 += 1
            path_wnid = wnid_in_path(str(sample.get("path_img", "")))
            if path_wnid is not None and path_wnid != ID_TO_WNID[class_id]:
                wnid_mismatch += 1

    total = len(samples)
    print(f"samples: {total}")
    if meta.get("source") is not None:
        print(f"source: {meta.get('source')}  prep: {meta.get('prep', '(absent)')}")
    print(f"raw acc1: {raw1}/{total} = {raw1 / total:.6f}")
    print(f"raw acc5: {raw5}/{total} = {raw5 / total:.6f}")
    meta_acc1 = meta.get("acc1")
    meta_acc5 = meta.get("acc5")
    if isinstance(meta_acc1, (int, float)) and isinstance(meta_acc5, (int, float)):
        print(f"meta acc1: {float(meta_acc1):.6f}  meta acc5: {float(meta_acc5):.6f}")
    print(f"mapped acc1: {mapped1}/{total} = {mapped1 / total:.6f}")
    print(f"mapped acc5: {mapped5}/{total} = {mapped5 / total:.6f}")
    print("per-class mapped acc1:")
    for class_id, wnid, imagenet_id in IMAGENETTE:
        count = per_n[class_id]
        hit = per_ok[class_id]
        if count == 0:
            print(f"  {class_id} {wnid} -> {imagenet_id:3d}  0/0")
        else:
            print(f"  {class_id} {wnid} -> {imagenet_id:3d}  {hit}/{count} = {hit / count:.3f}")
    if unknown:
        print(f"class_id outside 0..9: {unknown} (counted wrong)")
    if wnid_mismatch:
        print(f"path wnid disagrees with class_id: {wnid_mismatch}")
    if pred_mismatch:
        print(f"pred_id disagrees with argmax: {pred_mismatch}")
    return 0


def argmax_index(logits: list[float]) -> int:
    best_index = 0
    best_value = logits[0]
    for index, logit in enumerate(logits):
        if logit > best_value:
            best_value = logit
            best_index = index
    return best_index


if __name__ == "__main__":
    raise SystemExit(main())
