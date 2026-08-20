"""比较两个 ONNX：I/O 契约必须一致，主输出数值须在阈值内。图结构仅打印。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def summarize_onnx(path: Path) -> dict:
    model = onnx.load(str(path))
    opset = [o.version for o in model.opset_import]
    sess = _session(path)
    return {
        "path": str(path),
        "ir_version": model.ir_version,
        "opset": opset,
        "n_nodes": len(model.graph.node),
        "ops": sorted({n.op_type for n in model.graph.node}),
        "inputs": [(i.name, list(i.shape), i.type) for i in sess.get_inputs()],
        "outputs": [(o.name, list(o.shape), o.type) for o in sess.get_outputs()],
    }


def print_summary(label: str, info: dict) -> None:
    print(f"[{label}] {info['path']}")
    print(f"  ir={info['ir_version']} opset={info['opset']} nodes={info['n_nodes']}")
    print(f"  ops={info['ops']}")
    for name, dims, dtype in info["inputs"]:
        print(f"  in  {name} {dims} dtype={dtype}")
    for name, dims, dtype in info["outputs"]:
        print(f"  out {name} {dims} dtype={dtype}")


def check_io(a: dict, b: dict) -> list[str]:
    errors: list[str] = []
    if len(a["inputs"]) != len(b["inputs"]):
        errors.append(f"输入个数 {len(a['inputs'])} vs {len(b['inputs'])}")
    if len(a["outputs"]) != len(b["outputs"]):
        errors.append(f"输出个数 {len(a['outputs'])} vs {len(b['outputs'])}")
    for side, key in (("输入", "inputs"), ("输出", "outputs")):
        for i, (left, right) in enumerate(zip(a[key], b[key])):
            n1, d1, t1 = left
            n2, d2, t2 = right
            if n1 != n2:
                errors.append(f"{side}[{i}] 名称 {n1!r} vs {n2!r}")
            if d1 != d2:
                errors.append(f"{side}[{i}] {n1} 形状 {d1} vs {d2}")
            if t1 != t2:
                errors.append(f"{side}[{i}] {n1} dtype {t1} vs {t2}")
    return errors


def load_image_nchw(path: Path, height: int, width: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def make_dummy(shape: tuple[int, int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def run_onnx(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sess = _session(path)
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    return dict(zip(names, outs))


def pick_main_output(outs: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    if "output0" in outs:
        return "output0", outs["output0"]
    name = next(iter(outs))
    return name, outs[name]


def compare_arrays(
    ref: np.ndarray,
    other: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict:
    if ref.shape != other.shape:
        raise ValueError(f"输出形状 {ref.shape} vs {other.shape}")
    delta = np.abs(ref.astype(np.float64) - other.astype(np.float64))
    denom = np.abs(ref.astype(np.float64)) + 1e-12
    rel = delta / denom
    ok = bool(np.allclose(other, ref, atol=atol, rtol=rtol))
    stats = {
        "shape": tuple(ref.shape),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "ok": ok,
    }
    if ref.ndim == 3 and ref.shape[1] >= 5:
        box = delta[:, :4, :]
        cls = delta[:, 4:, :]
        stats["box_max_abs"] = float(box.max())
        stats["cls_max_abs"] = float(cls.max())
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="比较两个 ONNX 的 I/O 与主输出数值")
    p.add_argument("ref", type=Path, help="参考模型，如官方 yolov8n.onnx")
    p.add_argument("other", type=Path, help="待比较模型，如 yolov8_n.onnx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image", type=Path, default=None, help="可选；不传则用固定 seed 随机输入")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.ref.is_file():
        print(f"找不到参考模型: {args.ref}", file=sys.stderr)
        return 2
    if not args.other.is_file():
        print(f"找不到待比较模型: {args.other}", file=sys.stderr)
        return 2

    ref_info = summarize_onnx(args.ref)
    other_info = summarize_onnx(args.other)
    print_summary("ref", ref_info)
    print_summary("other", other_info)

    io_errors = check_io(ref_info, other_info)
    if io_errors:
        print("I/O 不一致:")
        for e in io_errors:
            print(f"  - {e}")
        return 1
    if not ref_info["inputs"]:
        print("参考模型没有可用输入（可能被 simplify 折成常量图）", file=sys.stderr)
        return 2

    in_name, in_dims, _ = ref_info["inputs"][0]
    if len(in_dims) != 4 or any(not isinstance(d, int) or int(d) <= 0 for d in in_dims):
        print(f"输入形状须为静态 [N,C,H,W]，当前: {in_dims}", file=sys.stderr)
        return 2
    n, c, h, w = (int(x) for x in in_dims)
    if args.image is not None:
        if not args.image.is_file():
            print(f"找不到图片: {args.image}", file=sys.stderr)
            return 2
        x = load_image_nchw(args.image, h, w)
        if x.shape[1] != c:
            print(f"图片通道 {x.shape[1]} 与模型 {c} 不一致", file=sys.stderr)
            return 2
        if n != 1:
            print(f"--image 仅支持 batch=1，模型 batch={n}", file=sys.stderr)
            return 2
    else:
        x = make_dummy((n, c, h, w), args.seed)

    feeds = {in_name: x}
    ref_outs = run_onnx(args.ref, feeds)
    other_outs = run_onnx(args.other, feeds)
    ref_name, ref_arr = pick_main_output(ref_outs)
    other_name, other_arr = pick_main_output(other_outs)
    if ref_name != other_name:
        print(f"主输出名称 {ref_name!r} vs {other_name!r}")
        return 1

    stats = compare_arrays(ref_arr, other_arr, atol=args.atol, rtol=args.rtol)
    print(f"主输出 {ref_name} {stats['shape']}")
    print(
        f"  max_abs={stats['max_abs']:.6e} mean_abs={stats['mean_abs']:.6e} "
        f"max_rel={stats['max_rel']:.6e}"
    )
    if "box_max_abs" in stats:
        print(
            f"  box_max_abs={stats['box_max_abs']:.6e} "
            f"cls_max_abs={stats['cls_max_abs']:.6e}"
        )
    print(f"  allclose atol={args.atol} rtol={args.rtol} -> {'PASS' if stats['ok'] else 'FAIL'}")
    return 0 if stats["ok"] else 1
