#!/usr/bin/env python3
"""MobileNetV2：比对板端 NPZ 与本目录 ONNX dump。

shape 必须已经一致。脚本不做转置。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MobileNetV2 board and ONNX float32 NPZ dumps")
    p.add_argument("--board", required=True, help="Board dump NPZ from img_tensor_dump_sigmastar")
    p.add_argument("--onnx", required=True, help="ONNX dump NPZ from eval_onnx_dump.py")
    return p.parse_args()


def load_meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta" not in npz.files:
        raise SystemExit("NPZ is missing meta")
    raw = np.asarray(npz["meta"]).tobytes()
    return json.loads(raw.decode("utf-8"))


def output_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    keys: list[str] = []
    index = 0
    while f"output_{index}" in npz.files:
        keys.append(f"output_{index}")
        index += 1
    return keys


def metrics(board: np.ndarray, onnx: np.ndarray) -> dict[str, float]:
    a = np.asarray(board, dtype=np.float64).reshape(-1)
    b = np.asarray(onnx, dtype=np.float64).reshape(-1)
    diff = a - b
    abs_diff = np.abs(diff)
    denom = float(np.mean(np.abs(b))) + 1e-12
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        cosine = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return {
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine,
        "p99": float(np.quantile(abs_diff, 0.99)),
        "rel": float(np.mean(abs_diff) / denom),
    }


def main() -> int:
    args = parse_args()
    board_path = Path(args.board)
    onnx_path = Path(args.onnx)
    if not board_path.is_file() or not onnx_path.is_file():
        print("dump file not found", file=sys.stderr)
        return 1

    board_npz = np.load(board_path)
    onnx_npz = np.load(onnx_path)
    board_meta = load_meta(board_npz)
    onnx_meta = load_meta(onnx_npz)

    board_sha = board_meta.get("input_sha256")
    onnx_sha = onnx_meta.get("input_sha256")
    if board_sha != onnx_sha:
        print("input_sha256 mismatch")
        print(f"  board {board_sha}")
        print(f"  onnx  {onnx_sha}")
        return 1

    if bool(board_meta.get("resized")) != bool(onnx_meta.get("resized")):
        print(
            "resized flag differs: "
            f"board={board_meta.get('resized')} onnx={onnx_meta.get('resized')}"
        )
        return 1
    if (board_meta.get("model_w"), board_meta.get("model_h")) != (
        onnx_meta.get("model_w"),
        onnx_meta.get("model_h"),
    ):
        print(
            "model input size differs: "
            f"board={board_meta.get('model_w')}x{board_meta.get('model_h')} "
            f"onnx={onnx_meta.get('model_w')}x{onnx_meta.get('model_h')}"
        )
        return 1

    board_keys = output_keys(board_npz)
    onnx_keys = output_keys(onnx_npz)
    if board_keys != onnx_keys or not board_keys:
        print(f"output keys differ: board={board_keys} onnx={onnx_keys}")
        return 1

    mismatched = False
    for key in board_keys:
        board_arr = np.asarray(board_npz[key])
        onnx_arr = np.asarray(onnx_npz[key])
        if board_arr.shape != onnx_arr.shape:
            print(f"{key} shape mismatch")
            print(f"  board {tuple(int(d) for d in board_arr.shape)}")
            print(f"  onnx  {tuple(int(d) for d in onnx_arr.shape)}")
            mismatched = True
            continue
        stats = metrics(board_arr, onnx_arr)
        print(
            f"{key} shape={tuple(int(d) for d in board_arr.shape)} "
            f"max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} "
            f"rmse={stats['rmse']:.6g} cosine={stats['cosine']:.6g} "
            f"p99={stats['p99']:.6g} rel={stats['rel']:.6g}"
        )
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
