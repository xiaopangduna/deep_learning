#!/usr/bin/env python3
"""按 CSV 跑分类 ONNX，写成与板端 classifier_dump_sigmastar 相同的 JSON。

不写 --box-scale，或值 <= 0：整图。大于 0 且 CSV 有 cx,cy,w,h：先按归一化框外扩裁切，
再做下面的几何。框列不全时退回整图。裁切用 float32，与板端 crop_cxcywh 一致。

几何对齐板端 --prep：

  imagenet（默认）  短边缩到 floor(min(H,W)/0.875)，再中心裁到模型尺寸。224 就是短边 256、中心裁 224
  letterbox         保持宽高比，不足处填 114
  resize            直接 INTER_LINEAR 到模型 H×W

像素公式是 (像素 - mean) / std。不写 --mean 和 --std 时用 RGB 的
123.68,116.28,103.53 和 58.395,57.12,57.375，与 input_config.ini 一致。
两边都写 none 时保持像素原值。

summary.acc1 / acc5 只比较 CSV 的 class_id 和 pred_id。原始 id 到训练 id 的对应不在这里做。
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
    p.add_argument(
        "--box-scale",
        type=float,
        default=None,
        help="Crop expand factor. >0 and CSV has cx,cy,w,h: crop before --prep. Omit or <=0: whole image",
    )
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


def load_rows(csv_path: Path) -> tuple[list[tuple[str, int, tuple[float, float, float, float] | None]], bool]:
    rows: list[tuple[str, int, tuple[float, float, float, float] | None]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"path_img", "class_id"}.issubset(set(reader.fieldnames)):
            raise SystemExit(f"csv missing required columns (path_img,class_id): {csv_path}")
        names = set(reader.fieldnames)
        box_names = {"cx", "cy", "w", "h"}
        present = box_names & names
        has_box = present == box_names
        if present and not has_box:
            print(f"csv missing some of cx,cy,w,h, using the whole image: {csv_path}", file=sys.stderr)
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
            box: tuple[float, float, float, float] | None = None
            if has_box:
                try:
                    box = (
                        float(np.float32((row.get("cx") or "").strip())),
                        float(np.float32((row.get("cy") or "").strip())),
                        float(np.float32((row.get("w") or "").strip())),
                        float(np.float32((row.get("h") or "").strip())),
                    )
                except ValueError:
                    print(f"csv line {line_no}: box is not numeric, skip", file=sys.stderr)
                    continue
            rows.append((path_img, class_id, box))
    return rows, has_box


def crop_cxcywh(
    src: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    box_scale: float,
) -> np.ndarray:
    """归一化 cxcywh 外扩后裁切，至少 1 像素。运算用 float32，对齐板端 crop_cxcywh。"""
    height, width = src.shape[:2]
    width_f = np.float32(width)
    height_f = np.float32(height)
    scale = np.float32(box_scale)
    abs_w = np.maximum(np.float32(w) * width_f * scale, np.float32(1.0))
    abs_h = np.maximum(np.float32(h) * height_f * scale, np.float32(1.0))
    abs_cx = np.float32(cx) * width_f
    abs_cy = np.float32(cy) * height_f
    half = np.float32(2.0)
    x1 = int(math.floor(float(abs_cx - abs_w / half)))
    y1 = int(math.floor(float(abs_cy - abs_h / half)))
    x2 = int(math.ceil(float(abs_cx + abs_w / half)))
    y2 = int(math.ceil(float(abs_cy + abs_h / half)))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return np.ascontiguousarray(src[y1:y2, x1:x2])


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


def static_output(shape: list[Any]) -> tuple[int, list[int]] | None:
    dims: list[int] = []
    numel = 1
    for dim in shape:
        if not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
            return None
        dims.append(int(dim))
        numel *= int(dim)
    if numel <= 0:
        return None
    return numel, dims


def indented(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")


class DumpWriter:
    """先写 meta，每条成功样本追加，最后写 summary。"""

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.staging = Path(str(out_path) + ".writing")
        self.handle = None
        self.written = 0

    def start(self, meta: dict[str, Any]) -> None:
        if self.handle is not None:
            return
        self.handle = self.staging.open("w", encoding="utf-8")
        self.handle.write('{\n  "meta": ')
        self.handle.write(indented(meta))
        self.handle.write(',\n  "samples": [')
        self.handle.flush()

    def append(self, sample: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("dump header was not started")
        if self.written:
            self.handle.write(",")
        self.handle.write("\n    ")
        self.handle.write(json.dumps(sample, ensure_ascii=False))
        self.written += 1
        self.handle.flush()

    def finish(self, summary: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("dump header was not started")
        try:
            if self.written == 0:
                self.handle.write('],\n  "summary": ')
            else:
                self.handle.write('\n  ],\n  "summary": ')
            self.handle.write(indented(summary))
            self.handle.write("\n}\n")
            self.handle.flush()
        except Exception:
            self.handle.close()
            self.handle = None
            print(f"partial dump kept at {self.staging}", file=sys.stderr)
            raise
        self.handle.close()
        self.handle = None
        try:
            self.staging.replace(self.out_path)
        except OSError as exc:
            raise SystemExit(f"failed to rename {self.staging} -> {self.out_path}: {exc}") from exc

    def close_partial(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


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
    rows, has_box = load_rows(csv_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    crop = args.box_scale is not None and args.box_scale > 0.0 and has_box
    if args.box_scale is not None and args.box_scale > 0.0 and not has_box:
        print("--box-scale > 0 but csv has no cx,cy,w,h, using the whole image", file=sys.stderr)

    out_path = Path(args.out) if args.out else Path(model_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_dir = csv_path.parent
    read_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

    ok_count = 0
    skip_count = 0
    correct1 = 0
    correct5 = 0
    num_classes = 0
    output_shape: list[int] = []
    top_k = 0
    spec = static_output(list(onnx_out.shape))
    if spec is not None:
        num_classes, output_shape = spec
        top_k = min(TOP_K_MAX, num_classes)

    def build_meta() -> dict[str, Any]:
        meta: dict[str, Any] = {
            "format_version": 2,
            "source": "onnx",
            "model": args.model,
            "csv": args.csv,
            "mode": "box" if crop else "image",
        }
        if crop:
            meta["box_scale"] = float(args.box_scale)
        meta["num_classes"] = num_classes
        meta["output_shape"] = output_shape
        meta["model_h"] = model_h
        meta["model_w"] = model_w
        meta["prep"] = args.prep
        if args.prep == "letterbox":
            meta["letterbox_fill"] = LETTERBOX_FILL
        meta["mean"] = mean_list
        meta["std"] = std_list
        meta["top_k"] = top_k
        return meta

    print(f"provider: {provider}")
    print(f"mode: {'box' if crop else 'image'}")
    if crop:
        print(f"box-scale: {args.box_scale}")
    print(f"prep: {args.prep}  input: {layout} {channels}x{model_h}x{model_w}")
    if mean is None:
        print("norm: none")
    else:
        print(f"norm: mean={mean_list} std={std_list}")
    print(f"rows: {len(rows)}")

    writer = DumpWriter(out_path)
    try:
        if num_classes > 0:
            writer.start(build_meta())
        for index, (path_img, class_id, box) in enumerate(rows):
            if crop and box is not None and not all(math.isfinite(value) for value in box):
                print(f"[{index}] skip non-finite box path={path_img}", file=sys.stderr)
                skip_count += 1
                continue
            image_path = Path(path_img)
            if not image_path.is_absolute():
                image_path = csv_dir / path_img
            image = cv2.imread(str(image_path), read_flag)
            if image is None:
                print(f"[{index}] failed to read: {image_path}", file=sys.stderr)
                skip_count += 1
                continue
            region = image
            if crop and box is not None:
                region = crop_cxcywh(image, box[0], box[1], box[2], box[3], args.box_scale)
            fitted = fit_model_canvas(region, model_w, model_h, args.prep)
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
                writer.start(build_meta())
            elif int(logits.size) != num_classes:
                print(f"[{index}] logit length {logits.size} != {num_classes}", file=sys.stderr)
                skip_count += 1
                continue
            pred_id = int(np.argmax(logits))
            sample: dict[str, Any] = {
                "row": index,
                "path_img": path_img,
                "class_id": class_id,
            }
            if crop and box is not None:
                sample["cx"] = box[0]
                sample["cy"] = box[1]
                sample["w"] = box[2]
                sample["h"] = box[3]
            sample["logits"] = [float(item) for item in logits]
            sample["pred_id"] = pred_id
            writer.append(sample)
            if pred_id == class_id:
                correct1 += 1
            if in_top_k(logits, class_id, top_k):
                correct5 += 1
            ok_count += 1
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"  progress: {index + 1} / {len(rows)} (ok={ok_count} skip={skip_count})")
    except Exception:
        writer.close_partial()
        if writer.staging.is_file():
            print(f"partial dump kept at {writer.staging}", file=sys.stderr)
        raise

    if writer.handle is None:
        writer.start(build_meta())
    if ok_count == 0:
        acc1: float | None = None
        acc5: float | None = None
    else:
        acc1 = correct1 / ok_count
        acc5 = correct5 / ok_count
    writer.finish(
        {
            "ok": ok_count,
            "skip": skip_count,
            "acc1": acc1,
            "acc5": acc5,
        }
    )

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
