#!/usr/bin/env python3
"""MobileNetV2：一张图跑 ONNX，写成与板端 img_tensor_dump 相同的 NPZ。

只服务本目录的分类模型。图应已是模型输入尺寸；尺寸不同时才 INTER_LINEAR。
输出按 Runtime 的顺序存 float32，不做转置。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump one MobileNetV2 ONNX inference to a float32 NPZ"
    )
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", required=True, help="One image, preferably already model HxW")
    p.add_argument("--out", default="dump_onnx.npz", help="Output NPZ path")
    p.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Multiply uint8 pixels after cast to float32 (1 keeps 0-255, 1/255 matches gray std_value=255)",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_dim(value: Any, what: str) -> int:
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    raise SystemExit(f"ONNX input {what} is not a fixed positive size: {value}")


def input_layout(shape: list[Any]) -> tuple[str, int, int, int]:
    """Return layout, C, H, W for a 4D ONNX input."""
    if len(shape) != 4:
        raise SystemExit(f"ONNX input rank {len(shape)} is unsupported, expect 4D NCHW or NHWC")
    c_nchw = shape[1]
    c_nhwc = shape[3]
    nchw = isinstance(c_nchw, (int, np.integer)) and int(c_nchw) in (1, 3)
    nhwc = isinstance(c_nhwc, (int, np.integer)) and int(c_nhwc) in (1, 3)
    if nchw:
        return (
            "NCHW",
            int(c_nchw),
            _fixed_dim(shape[2], "H"),
            _fixed_dim(shape[3], "W"),
        )
    if nhwc:
        return (
            "NHWC",
            int(c_nhwc),
            _fixed_dim(shape[1], "H"),
            _fixed_dim(shape[2], "W"),
        )
    raise SystemExit(f"cannot tell NCHW vs NHWC from ONNX input shape {shape}")


def prepare_input(
    image_path: Path,
    layout: str,
    channels: int,
    model_h: int,
    model_w: int,
    value_scale: float,
) -> tuple[np.ndarray, int, int, bool, str]:
    import cv2

    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise SystemExit(f"failed to decode image: {image_path}")
    image_h, image_w = image.shape[:2]
    resized = False
    if image_w != model_w or image_h != model_h:
        image = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        resized = True
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color = "RGB"
    else:
        color = "GRAY"

    array = image.astype(np.float32) * np.float32(value_scale)
    if layout == "NCHW":
        if array.ndim == 2:
            array = array[None, None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))[None, :, :, :]
    elif layout == "NHWC":
        if array.ndim == 2:
            array = array[None, :, :, None]
        else:
            array = array[None, :, :, :]
    else:
        raise SystemExit(f"unsupported layout {layout}")
    return np.ascontiguousarray(array), image_w, image_h, resized, color


def main() -> int:
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is required: pip install onnxruntime", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not image_path.is_file():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        print(f"expected 1 ONNX input, got {len(inputs)}", file=sys.stderr)
        return 1
    onnx_in = inputs[0]
    layout, channels, model_h, model_w = input_layout(list(onnx_in.shape))
    tensor, image_w, image_h, resized, color = prepare_input(
        image_path, layout, channels, model_h, model_w, args.value_scale
    )

    output_metas = sess.get_outputs()
    names = [item.name for item in output_metas]
    values = sess.run(names, {onnx_in.name: tensor})

    arrays: dict[str, np.ndarray] = {}
    outputs_meta: list[dict[str, Any]] = []
    for index, (name, value) in enumerate(zip(names, values)):
        src = np.asarray(value)
        stored = np.ascontiguousarray(src, dtype=np.float32)
        key = f"output_{index}"
        arrays[key] = stored
        outputs_meta.append(
            {
                "key": key,
                "name": name,
                "shape": [int(d) for d in stored.shape],
                "src_dtype": str(src.dtype),
                "fscalar": 1.0,
                "zero_point": 0,
                "numel": int(stored.size),
            }
        )
        print(f"  {key} name={name} shape={list(stored.shape)} src={src.dtype}")

    meta = {
        "format_version": 1,
        "source": "onnx",
        "model": model_path.name,
        "image": image_path.name,
        "input_sha256": file_sha256(image_path),
        "resized": bool(resized),
        "image_w": int(image_w),
        "image_h": int(image_h),
        "model_w": int(model_w),
        "model_h": int(model_h),
        "value_scale": float(args.value_scale),
        "input_layout": layout,
        "input_color": color,
        "input_name": onnx_in.name,
        "dequant": "identity",
        "stored_dtype": "float32",
        "outputs": outputs_meta,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    arrays["meta"] = np.frombuffer(meta_bytes, dtype=np.uint8).copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
