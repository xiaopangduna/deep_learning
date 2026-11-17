#!/usr/bin/env python3
"""比较两个图片输入 ONNX 的输出。不传图片则随机生成，传入则缩放。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lovely_deep_learning.export.compare import ATOL, RTOL, OnnxCompareResult, compare_onnx


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两个图片输入 ONNX 的输出")
    parser.add_argument("ref", type=Path, help="参考 ONNX")
    parser.add_argument("other", type=Path, help="待比较 ONNX")
    parser.add_argument("--image", type=Path, default=None, help="不传则随机生成，传了则读取并缩放")
    return parser.parse_args(argv)


def _shape(shape: list[int | None] | tuple[int | None, ...]) -> str:
    return "[" + ", ".join("null" if dim is None else str(dim) for dim in shape) + "]"


def format_report(result: OnnxCompareResult) -> str:
    lines = [f"atol={ATOL} rtol={RTOL}"]
    for label, info in (("ref", result.ref), ("other", result.other)):
        lines.append(f"{label}  {info.path}  nodes={info.nodes} opset={list(info.opset)}")
        lines.append(f"  in  {info.input.name} {_shape(info.input.shape)} {info.input.dtype}")
        for index, item in enumerate(info.outputs):
            lines.append(f"  out[{index}] {item.name} {_shape(item.shape)} {item.dtype}")
    if result.feed is None:
        lines.append("feed  未运行")
    else:
        source = "random" if result.image is None else result.image
        lines.append(f"feed  {source}")
        lines.append(f"  {_shape(result.feed.shape)} {result.feed.dtype}")
    for item in result.errors:
        lines.append(f"error  {item}")
    for item in result.outputs:
        flag = "PASS" if item.ok else "FAIL"
        lines.append(
            f"out[{item.index}] {item.ref_name} vs {item.other_name}  {_shape(item.shape)}  "
            f"max_abs={item.max_abs:.6e} mean_abs={item.mean_abs:.6e}  {flag}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = compare_onnx(args.ref, args.other, image=args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
