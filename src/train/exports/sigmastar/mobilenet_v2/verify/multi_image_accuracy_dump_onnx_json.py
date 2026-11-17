#!/usr/bin/env python3
"""按 CSV 跑分类 ONNX，写成与板端 classifier_dump_sigmastar 相同的 JSON。

整图，不裁框。几何对齐板端 --prep：

  imagenet（默认）  短边缩到 floor(min(H,W)/0.875)，再中心裁到模型尺寸。224 就是短边 256、中心裁 224
  letterbox         保持宽高比，不足处填 114
  resize            直接 INTER_LINEAR 到模型 H×W

像素公式是 (像素 - mean) / std。不写 --mean 和 --std 时用 RGB 的
123.68,116.28,103.53 和 58.395,57.12,57.375，与 input_config.ini 一致。
两边都写 none 时保持像素原值。

推理优先用 CUDA。先 preload uv 装进 site-packages 的 CUDA 12 / cuDNN 9，
这样不必依赖 shell 里的 LD_LIBRARY_PATH。CUDA 建不起来就退回 CPU。

meta.acc1 / acc5 只比较 CSV 的 class_id 和 pred_id，不做 Imagenette 到 ImageNet 的映射。
映射后的精度用 multi_image_accuracy_score_imagenette_json.py。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from input_norm import resolve_norm

LETTERBOX_FILL = 114
TOP_K_MAX = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump a CSV through a classifier ONNX to board-format JSON")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--csv", required=True, help="CSV with path_img,class_id. path_img is relative to the CSV directory")
    p.add_argument("--out", default=None, help="Output JSON. Default is the ONNX file stem with .json")
    p.add_argument(
        "--prep",
        choices=("imagenet", "letterbox", "resize"),
        default="imagenet",
        help="Spatial prep, same names as the board app. Default: imagenet",
    )
    p.add_argument(
        "--mean",
        default=None,
        help="Per-channel mean, comma-separated. Omit with --std to use 123.68,116.28,103.53. 'none' keeps raw pixels",
    )
    p.add_argument(
        "--std",
        default=None,
        help="Per-channel std, comma-separated. Omit with --mean to use 58.395,57.12,57.375. 'none' keeps raw pixels",
    )
    p.add_argument("--limit", type=int, default=None, help="Only the first N CSV rows")
    return p.parse_args()


def lround(value: float) -> int:
    """std::lround: half away from zero. Python round is bankers rounding."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _fixed_dim(value: Any, what: str) -> int:
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    raise SystemExit(f"ONNX input {what} is not a fixed positive size: {value}")


def input_layout(shape: list[Any]) -> tuple[str, int, int, int]:
    if len(shape) != 4:
        raise SystemExit(f"ONNX input rank {len(shape)} is unsupported, expect 4D NCHW or NHWC")
    c_nchw = shape[1]
    c_nhwc = shape[3]
    nchw = isinstance(c_nchw, (int, np.integer)) and int(c_nchw) in (1, 3)
    nhwc = isinstance(c_nhwc, (int, np.integer)) and int(c_nhwc) in (1, 3)
    if nchw:
        return "NCHW", int(c_nchw), _fixed_dim(shape[2], "H"), _fixed_dim(shape[3], "W")
    if nhwc:
        return "NHWC", int(c_nhwc), _fixed_dim(shape[1], "H"), _fixed_dim(shape[2], "W")
    raise SystemExit(f"cannot tell NCHW vs NHWC from ONNX input shape {shape}")


def letterbox(src: np.ndarray, dst_w: int, dst_h: int, fill: int) -> np.ndarray:
    import cv2

    src_h, src_w = src.shape[:2]
    ratio = min(dst_h / float(src_h), dst_w / float(src_w))
    resize_w = lround(src_w * ratio)
    resize_h = lround(src_h * ratio)
    dw = (dst_w - resize_w) / 2.0
    dh = (dst_h - resize_h) / 2.0
    top = lround(dh - 0.1)
    bottom = lround(dh + 0.1)
    left = lround(dw - 0.1)
    right = lround(dw + 0.1)
    resized = cv2.resize(src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    value: int | tuple[int, int, int] = fill if src.ndim == 2 else (fill, fill, fill)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
    )
    if padded.shape[1] == dst_w and padded.shape[0] == dst_h:
        return padded
    if src.ndim == 2:
        exact = np.full((dst_h, dst_w), fill, dtype=padded.dtype)
    else:
        exact = np.full((dst_h, dst_w, padded.shape[2]), fill, dtype=padded.dtype)
    copy_h = min(padded.shape[0], dst_h)
    copy_w = min(padded.shape[1], dst_w)
    exact[:copy_h, :copy_w] = padded[:copy_h, :copy_w]
    return exact


def resize_shorter_side(src: np.ndarray, shorter: int) -> np.ndarray:
    import cv2

    src_h, src_w = src.shape[:2]
    if src_w <= 0 or src_h <= 0 or shorter <= 0 or min(src_w, src_h) == shorter:
        return src
    if src_w < src_h:
        new_w = shorter
        new_h = max(1, lround(src_h * shorter / float(src_w)))
    else:
        new_h = shorter
        new_w = max(1, lround(src_w * shorter / float(src_h)))
    return cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(src: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    import cv2

    src_h, src_w = src.shape[:2]
    if src_w == crop_w and src_h == crop_h:
        return src
    if src_w < crop_w or src_h < crop_h:
        return cv2.resize(src, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    x = lround((src_w - crop_w) / 2.0)
    y = lround((src_h - crop_h) / 2.0)
    return np.ascontiguousarray(src[y : y + crop_h, x : x + crop_w])


def fit_model_canvas(src: np.ndarray, model_w: int, model_h: int, prep: str) -> np.ndarray:
    import cv2

    if prep == "resize":
        if src.shape[1] == model_w and src.shape[0] == model_h:
            return src
        return cv2.resize(src, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    if prep == "imagenet":
        crop = min(model_w, model_h)
        shorter = int(math.floor(float(np.float32(crop) / np.float32(0.875))))
        return center_crop(resize_shorter_side(src, shorter), model_w, model_h)
    return letterbox(src, model_w, model_h, LETTERBOX_FILL)


def load_rows(csv_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"path_img", "class_id"}.issubset(set(reader.fieldnames)):
            raise SystemExit(f"csv missing required columns (path_img,class_id): {csv_path}")
        for line_no, row in enumerate(reader, start=2):
            path_img = (row.get("path_img") or "").strip()
            class_text = (row.get("class_id") or "").strip()
            if path_img == "" and class_text == "":
                continue
            if path_img == "" or class_text == "":
                print(f"csv line {line_no}: too few fields, skip", file=sys.stderr)
                continue
            try:
                class_id = int(class_text)
            except ValueError:
                print(f"csv line {line_no}: class_id {class_text!r} is not an int, skip", file=sys.stderr)
                continue
            rows.append((path_img, class_id))
    return rows


def in_top_k(logits: np.ndarray, class_id: int, top_k: int) -> bool:
    if class_id < 0 or class_id >= logits.shape[0] or top_k <= 0:
        return False
    target = float(logits[class_id])
    better = 0
    for index, logit in enumerate(logits):
        if index == class_id:
            continue
        value = float(logit)
        if value > target or (value == target and index < class_id):
            better += 1
            if better >= top_k:
                return False
    return True


def to_model_input(
    image: np.ndarray,
    layout: str,
    channels: int,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    import cv2

    if channels == 3:
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        array = image.astype(np.float32)
    if mean is not None and std is not None:
        if array.ndim == 2:
            array = (array - mean[0]) / std[0]
        else:
            array = (array - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    if layout == "NCHW":
        if array.ndim == 2:
            array = array[None, None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))[None, :, :, :]
    else:
        if array.ndim == 2:
            array = array[None, :, :, None]
        else:
            array = array[None, :, :, :]
    return np.ascontiguousarray(array)


def make_session(model_path: Path):
    import onnxruntime as ort

    ort.preload_dlls()
    available = ort.get_available_providers()
    providers = [
        name
        for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if name in available
    ]
    if not providers:
        raise SystemExit(f"no supported provider in {available}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    used = session.get_providers()
    return session, used[0] if used else providers[0]


def write_dump(out_path: Path, meta: dict[str, Any], sample_path: Path, ok_count: int) -> None:
    staging = Path(str(out_path) + ".writing")
    with staging.open("w", encoding="utf-8") as handle, sample_path.open(encoding="utf-8") as samples:
        handle.write('{\n  "meta": ')
        meta_text = json.dumps(meta, indent=2, ensure_ascii=False)
        handle.write(meta_text.replace("\n", "\n  "))
        handle.write(',\n  "samples": [')
        if ok_count == 0:
            handle.write("]\n}\n")
            return
        handle.write("\n")
        written = 0
        for line in samples:
            if not line.strip():
                continue
            written += 1
            handle.write("    ")
            handle.write(line.rstrip("\n"))
            if written < ok_count:
                handle.write(",")
            handle.write("\n")
        handle.write("  ]\n}\n")
        if written != ok_count:
            raise SystemExit(f"sample count {written} != ok {ok_count}")
    staging.replace(out_path)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        return 1
    try:
        import cv2
    except ImportError:
        print("opencv is required", file=sys.stderr)
        return 1
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("onnxruntime is required", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    csv_path = Path(args.csv)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not csv_path.is_file():
        print(f"csv not found: {csv_path}", file=sys.stderr)
        return 1

    session, provider = make_session(model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        print(f"expected 1 ONNX input, got {len(inputs)}", file=sys.stderr)
        return 1
    if len(outputs) != 1:
        print(f"expected 1 ONNX output, got {len(outputs)}", file=sys.stderr)
        return 1
    onnx_in = inputs[0]
    onnx_out = outputs[0]
    layout, channels, model_h, model_w = input_layout(list(onnx_in.shape))
    mean_list, std_list = resolve_norm(args.mean, args.std, channels)
    mean = None if mean_list is None else np.asarray(mean_list, dtype=np.float32)
    std = None if std_list is None else np.asarray(std_list, dtype=np.float32)
    rows = load_rows(csv_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_path = Path(args.out) if args.out else Path(model_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = Path(str(out_path) + ".samples.tmp")
    csv_dir = csv_path.parent
    read_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

    ok_count = 0
    skip_count = 0
    correct1 = 0
    correct5 = 0
    num_classes = 0
    output_shape: list[int] = []
    top_k = 1

    print(f"provider: {provider}")
    print(f"prep: {args.prep}  input: {layout} {channels}x{model_h}x{model_w}")
    if mean is None:
        print("norm: none")
    else:
        print(f"norm: mean={mean_list} std={std_list}")
    print(f"rows: {len(rows)}")

    try:
        with sample_path.open("w", encoding="utf-8") as sample_file:
            for index, (path_img, class_id) in enumerate(rows):
                image_path = Path(path_img)
                if not image_path.is_absolute():
                    image_path = csv_dir / path_img
                image = cv2.imread(str(image_path), read_flag)
                if image is None:
                    print(f"[{index}] failed to read: {image_path}", file=sys.stderr)
                    skip_count += 1
                    continue
                fitted = fit_model_canvas(image, model_w, model_h, args.prep)
                if fitted.shape[0] != model_h or fitted.shape[1] != model_w:
                    print(f"[{index}] prep size {fitted.shape} != {model_h}x{model_w}", file=sys.stderr)
                    skip_count += 1
                    continue
                tensor = to_model_input(fitted, layout, channels, mean, std)
                value = session.run([onnx_out.name], {onnx_in.name: tensor})[0]
                logits = np.asarray(value, dtype=np.float32).reshape(-1)
                if logits.size == 0 or not np.isfinite(logits).all():
                    print(f"[{index}] non-finite logits: {path_img}", file=sys.stderr)
                    skip_count += 1
                    continue
                if num_classes == 0:
                    num_classes = int(logits.size)
                    output_shape = [int(dim) for dim in np.asarray(value).shape]
                    top_k = min(TOP_K_MAX, num_classes)
                elif int(logits.size) != num_classes:
                    print(f"[{index}] logit length {logits.size} != {num_classes}", file=sys.stderr)
                    skip_count += 1
                    continue
                pred_id = int(np.argmax(logits))
                sample = {
                    "row": index,
                    "path_img": path_img,
                    "class_id": class_id,
                    "logits": [float(item) for item in logits],
                    "pred_id": pred_id,
                }
                sample_file.write(json.dumps(sample, ensure_ascii=False))
                sample_file.write("\n")
                if pred_id == class_id:
                    correct1 += 1
                if in_top_k(logits, class_id, top_k):
                    correct5 += 1
                ok_count += 1
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(f"  progress: {index + 1} / {len(rows)} (ok={ok_count} skip={skip_count})")
    except Exception:
        sample_path.unlink(missing_ok=True)
        raise

    if ok_count == 0:
        acc1: float | None = None
        acc5: float | None = None
    else:
        acc1 = correct1 / ok_count
        acc5 = correct5 / ok_count
    meta: dict[str, Any] = {
        "format_version": 1,
        "source": "onnx",
        "model": args.model,
        "csv": args.csv,
        "mode": "image",
        "num_classes": num_classes,
        "output_shape": output_shape,
        "model_h": model_h,
        "model_w": model_w,
        "prep": args.prep,
        "mean": mean_list,
        "std": std_list,
        "ok": ok_count,
        "skip": skip_count,
        "top_k": top_k if ok_count else min(TOP_K_MAX, num_classes),
        "acc1": acc1,
        "acc5": acc5,
    }
    if args.prep == "letterbox":
        meta["letterbox_fill"] = LETTERBOX_FILL
    try:
        write_dump(out_path, meta, sample_path, ok_count)
    finally:
        sample_path.unlink(missing_ok=True)

    if ok_count == 0:
        print("acc1: n/a")
        print("acc5: n/a")
        print(f"done: {out_path}")
        return 1
    print(f"acc1: {correct1}/{ok_count} = {acc1:.6f}")
    print(f"acc5: {correct5}/{ok_count} = {acc5:.6f} (top_k={top_k})")
    print(f"done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
