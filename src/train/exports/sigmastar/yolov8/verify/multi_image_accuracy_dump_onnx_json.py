#!/usr/bin/env python3
"""按检测 CSV 跑 YOLOv8 ONNX，写成可算 mAP 的 JSON。

CSV 需要 path_img 和 path_label_detect_yolo，路径相对 CSV 所在目录。标签是
YOLO 检测格式：每行 class_id cx cy w h，归一化到原图。文件缺失或为空当作没有目标。

几何是 Ultralytics LetterBox：保持宽高比，不足处填 114，框再映射回原图像素 xyxy。
像素公式是 (像素 - mean) / std。不写 --mean 和 --std 时用 0 和 255，与
input_config.ini 一致，也就是 RGB 除以 255。两边都写 none 时保持像素原值。

ONNX 须是检测导出、框已解码：output0 为 (1, 4+nc, A) 或 (1, A, 4+nc)，前 4 个通道是
letterbox 画布上的 cxcywh，其余是 sigmoid 类别分。末维为 6 的 (1, K, 6) 当成已经
做过 NMS 的 xyxy、score、class。

后处理与 YOLO.val 相同：--conf 默认 0.001，按类 NMS 的 IoU 默认 0.7，每图最多 300 框，
multi_label。JSON 只写 NMS 之后的框。mAP50-95 用这些框对原图 GT 积分。summary 里的
precision / recall 是 F1 最大的那个置信度。precision_curve 和 recall_curve 对置信度
0 到 1，IoU 固定 0.5。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

LETTERBOX_FILL = 114
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a detection CSV through YOLOv8 ONNX to mAP JSON")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with path_img,path_label_detect_yolo. Paths are relative to the CSV directory",
    )
    parser.add_argument("--out", default=None, help="Output JSON. Default is the ONNX file stem with .json")
    parser.add_argument(
        "--mean",
        default=None,
        help="Per-channel mean. Omit with --std to use 0. 'none' keeps raw pixels",
    )
    parser.add_argument(
        "--std",
        default=None,
        help="Per-channel std. Omit with --mean to use 255. 'none' keeps raw pixels",
    )
    parser.add_argument("--conf", type=float, default=0.001, help="Score kept for NMS and mAP. Default 0.001")
    parser.add_argument("--iou", type=float, default=0.7, help="Class-wise NMS IoU. Default 0.7")
    parser.add_argument("--max-det", type=int, default=300, help="Max boxes per image after NMS. Default 300")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N CSV rows")
    return parser.parse_args()


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


def _parse_floats(text: str, flag: str) -> list[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise SystemExit(f"{flag} is empty")
    values: list[float] = []
    for item in parts:
        try:
            values.append(float(item))
        except ValueError:
            raise SystemExit(f"{flag} value {item!r} is not a number or none") from None
    return values


def _expand(values: list[float], channels: int, flag: str) -> list[float]:
    if len(values) == 1:
        return values * channels
    if len(values) == channels:
        return values
    raise SystemExit(f"{flag} has {len(values)} values, model has {channels} channels")


def resolve_norm(
    mean_text: str | None,
    std_text: str | None,
    channels: int,
) -> tuple[list[float] | None, list[float] | None]:
    """YOLOv8 input_config.ini 是 mean=0、std=255。省略时用这组，不是 ImageNet。"""
    if mean_text is not None:
        mean_text = mean_text.strip()
    if std_text is not None:
        std_text = std_text.strip()
    if mean_text is None and std_text is None:
        mean_vals = [0.0]
        std_vals = [255.0]
    elif mean_text is None or std_text is None:
        raise SystemExit("--mean and --std must be passed together")
    elif mean_text.lower() == "none" and std_text.lower() == "none":
        return None, None
    elif mean_text.lower() == "none" or std_text.lower() == "none":
        raise SystemExit("--mean and --std must both be none, or both be numbers")
    else:
        mean_vals = _parse_floats(mean_text, "--mean")
        std_vals = _parse_floats(std_text, "--std")
    mean_vals = _expand(mean_vals, channels, "--mean")
    std_vals = _expand(std_vals, channels, "--std")
    if any(value == 0.0 for value in std_vals):
        raise SystemExit("--std contains 0")
    return mean_vals, std_vals


def letterbox(src: np.ndarray, dst_w: int, dst_h: int, fill: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """与 Ultralytics LetterBox(center=True, auto=False, scaleup=True) 相同。返回图、比例、左上 padding。"""
    import cv2

    src_h, src_w = src.shape[:2]
    ratio = min(dst_h / float(src_h), dst_w / float(src_w))
    resize_w = int(round(src_w * ratio))
    resize_h = int(round(src_h * ratio))
    dw = (dst_w - resize_w) / 2.0
    dh = (dst_h - resize_h) / 2.0
    if (src_w, src_h) != (resize_w, resize_h):
        src = cv2.resize(src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    value: int | tuple[int, int, int] = fill if src.ndim == 2 else (fill, fill, fill)
    padded = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return padded, ratio, (left, top)


def load_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"path_img", "path_label_detect_yolo"}.issubset(set(reader.fieldnames)):
            raise SystemExit(f"csv missing required columns (path_img,path_label_detect_yolo): {csv_path}")
        for line_no, row in enumerate(reader, start=2):
            path_img = (row.get("path_img") or "").strip()
            path_label = (row.get("path_label_detect_yolo") or "").strip()
            if path_img == "" and path_label == "":
                continue
            if path_img == "":
                print(f"csv line {line_no}: empty path_img, skip", file=sys.stderr)
                continue
            rows.append((path_img, path_label))
    return rows


def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    """YOLO 归一化 cxcywh → 原图像素 xyxy。返回 (N, 5)：class_id, x1, y1, x2, y2。"""
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    boxes: list[list[float]] = []
    text = label_path.read_text(encoding="utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            print(f"{label_path}:{line_no}: fewer than 5 fields, skip", file=sys.stderr)
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            print(f"{label_path}:{line_no}: not numeric, skip", file=sys.stderr)
            continue
        if class_id < 0 or not all(np.isfinite(value) for value in (cx, cy, bw, bh)):
            print(f"{label_path}:{line_no}: non-finite box, skip", file=sys.stderr)
            continue
        x1 = float(np.clip((cx - bw * 0.5) * width, 0.0, width))
        y1 = float(np.clip((cy - bh * 0.5) * height, 0.0, height))
        x2 = float(np.clip((cx + bw * 0.5) * width, 0.0, width))
        y2 = float(np.clip((cy + bh * 0.5) * height, 0.0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([float(class_id), x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


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


def prediction_layout(value: np.ndarray) -> tuple[str, np.ndarray, int]:
    """返回 kind、形如 (1, C, A) 的 raw 或 (1, K, 6) 的 nms，以及类别数。nms 的类别数为 0。"""
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or int(array.shape[0]) != 1:
        raise SystemExit(f"ONNX output shape {tuple(int(dim) for dim in array.shape)} is unsupported, expect (1, 4+nc, A)")
    dim1 = int(array.shape[1])
    dim2 = int(array.shape[2])
    if dim2 == 6 and dim1 <= 1000:
        return "nms", np.ascontiguousarray(array), 0
    if dim1 >= dim2:
        array = np.transpose(array, (0, 2, 1))
        dim1 = int(array.shape[1])
    if dim1 <= 4:
        raise SystemExit(f"ONNX output channels {dim1} have no class scores")
    return "raw", np.ascontiguousarray(array), dim1 - 4


def decode_dets(
    kind: str,
    prediction: np.ndarray,
    nc: int,
    conf: float,
    iou: float,
    max_det: int,
    model_hw: tuple[int, int],
    orig_hw: tuple[int, int],
    ratio: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """返回原图像素 (N, 6)：class_id, score, x1, y1, x2, y2。按分数从高到低。"""
    import torch
    from ultralytics.utils.ops import non_max_suppression, scale_boxes

    if kind == "raw":
        tensor = torch.from_numpy(np.ascontiguousarray(prediction.astype(np.float32, copy=False)))
        kept = non_max_suppression(
            tensor,
            conf_thres=conf,
            iou_thres=iou,
            nc=nc,
            multi_label=True,
            max_det=max_det,
            in_place=False,
            max_time_img=1.0e6,
        )[0]
    else:
        rows = prediction[0].astype(np.float32, copy=False)
        if rows.size == 0:
            kept = torch.zeros((0, 6))
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(rows))
            tensor = tensor[tensor[:, 4] > conf]
            if tensor.shape[0] > max_det:
                tensor = tensor[tensor[:, 4].argsort(descending=True)[:max_det]]
            kept = tensor
    if kept.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    boxes = scale_boxes(model_hw, kept[:, :4].clone(), orig_hw, ratio_pad=((ratio, ratio), pad))
    order = torch.argsort(kept[:, 4], descending=True)
    out = torch.cat((kept[order, 5:6], kept[order, 4:5], boxes[order]), dim=1)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """与 ObjectDetectMetric 相同：每个 IoU 阈值上，一个预测最多配一个 GT。"""
    correct = np.zeros((pred_classes.shape[0], IOU_THRESHOLDS.size), dtype=bool)
    iou_np = iou * (true_classes[:, None] == pred_classes)
    for index, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou_np >= threshold)).T
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return correct


def make_session(model_path: Path):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    available = ort.get_available_providers()
    providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
    if not providers:
        raise SystemExit(f"no supported provider in {available}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    used = session.get_providers()
    return session, used[0] if used else providers[0]


def indented(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")


def round_rows(rows: np.ndarray, with_score: bool) -> list[list[float]]:
    dumped: list[list[float]] = []
    for row in rows:
        if with_score:
            dumped.append(
                [
                    int(row[0]),
                    round(float(row[1]), 5),
                    round(float(row[2]), 3),
                    round(float(row[3]), 3),
                    round(float(row[4]), 3),
                    round(float(row[5]), 3),
                ]
            )
        else:
            dumped.append(
                [
                    int(row[0]),
                    round(float(row[1]), 3),
                    round(float(row[2]), 3),
                    round(float(row[3]), 3),
                    round(float(row[4]), 3),
                ]
            )
    return dumped


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


def empty_summary() -> dict[str, Any]:
    return {
        "map": None,
        "map50": None,
        "map75": None,
        "precision": None,
        "recall": None,
        "f1_confidence": None,
        "per_class": [],
        "confidence": [],
        "precision_curve": [],
        "recall_curve": [],
    }


def summarize(metric: Any) -> dict[str, Any]:
    from ultralytics.utils.metrics import smooth

    box = metric.box
    if len(box.all_ap) == 0:
        return empty_summary()
    f1_mean = smooth(np.asarray(box.f1_curve).mean(0), 0.1)
    f1_index = int(np.nanargmax(f1_mean))
    per_class = []
    for item in metric.summary(decimals=5):
        per_class.append(
            {
                "class_id": int(item["Class"]),
                "images": int(item["Images"]),
                "instances": int(item["Instances"]),
                "precision": float(item["Box-P"]),
                "recall": float(item["Box-R"]),
                "f1": float(item["Box-F1"]),
                "map50": float(item["mAP50"]),
                "map": float(item["mAP50-95"]),
            }
        )
    confidence = np.asarray(box.px, dtype=np.float64)
    precision_curve = np.asarray(box.p_curve, dtype=np.float64).mean(0)
    recall_curve = np.asarray(box.r_curve, dtype=np.float64).mean(0)
    return {
        "map": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1_confidence": float(confidence[f1_index]),
        "per_class": per_class,
        "confidence": [round(float(value), 6) for value in confidence],
        "precision_curve": [round(float(value), 6) for value in precision_curve],
        "recall_curve": [round(float(value), 6) for value in recall_curve],
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        return 1
    if not 0.0 <= args.conf <= 1.0:
        print("--conf must be in [0, 1]", file=sys.stderr)
        return 1
    if not 0.0 <= args.iou <= 1.0:
        print("--iou must be in [0, 1]", file=sys.stderr)
        return 1
    if args.max_det < 1:
        print("--max-det must be >= 1", file=sys.stderr)
        return 1
    try:
        import cv2
    except ImportError:
        print("opencv is required", file=sys.stderr)
        return 1
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnxruntime is required", file=sys.stderr)
        return 1
    try:
        import torch
        from ultralytics.utils.metrics import DetMetrics, box_iou
    except ImportError:
        print("torch and ultralytics are required", file=sys.stderr)
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
    csv_dir = csv_path.parent
    read_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    meta: dict[str, Any] = {
        "format_version": 1,
        "source": "onnx",
        "task": "detect",
        "model": args.model,
        "csv": args.csv,
        "model_h": model_h,
        "model_w": model_w,
        "prep": "letterbox",
        "letterbox_fill": LETTERBOX_FILL,
        "mean": mean_list,
        "std": std_list,
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "box_format": "xyxy",
        "coords": "original_image",
    }

    print(f"provider: {provider}")
    print(f"prep: letterbox  input: {layout} {channels}x{model_h}x{model_w}")
    if mean is None:
        print("norm: none")
    else:
        print(f"norm: mean={mean_list} std={std_list}")
    print(f"nms: conf={args.conf} iou={args.iou} max_det={args.max_det}")
    print(f"rows: {len(rows)}")

    ok_count = 0
    skip_count = 0
    nc = 0
    saw_target = False
    metric = DetMetrics(names={})
    writer = DumpWriter(out_path)
    try:
        writer.start(meta)
        for index, (path_img, path_label) in enumerate(rows):
            image_path = Path(path_img)
            if not image_path.is_absolute():
                image_path = csv_dir / path_img
            image = cv2.imread(str(image_path), read_flag)
            if image is None:
                print(f"[{index}] failed to read: {image_path}", file=sys.stderr)
                skip_count += 1
                continue
            orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
            fitted, ratio, pad = letterbox(image, model_w, model_h, LETTERBOX_FILL)
            if fitted.shape[0] != model_h or fitted.shape[1] != model_w:
                print(
                    f"[{index}] letterbox size {fitted.shape[0]}x{fitted.shape[1]} != {model_h}x{model_w}",
                    file=sys.stderr,
                )
                skip_count += 1
                continue
            tensor = to_model_input(fitted, layout, channels, mean, std)
            value = np.asarray(session.run([onnx_out.name], {onnx_in.name: tensor})[0])
            if not np.isfinite(value).all():
                print(f"[{index}] non-finite output: {path_img}", file=sys.stderr)
                skip_count += 1
                continue
            kind, prediction, pred_nc = prediction_layout(value)
            if kind == "raw":
                if nc == 0:
                    nc = pred_nc
                elif pred_nc != nc:
                    print(f"[{index}] class count {pred_nc} != {nc}", file=sys.stderr)
                    skip_count += 1
                    continue
            label_path = csv_dir / path_label if path_label and not Path(path_label).is_absolute() else Path(path_label)
            gt = load_gt(label_path, orig_w, orig_h) if path_label else np.zeros((0, 5), dtype=np.float32)
            dets = decode_dets(
                kind,
                prediction,
                nc if kind == "raw" else 0,
                args.conf,
                args.iou,
                args.max_det,
                (model_h, model_w),
                (orig_h, orig_w),
                ratio,
                pad,
            )
            if kind == "nms":
                observed = []
                if gt.shape[0]:
                    observed.append(int(gt[:, 0].max()))
                if dets.shape[0]:
                    observed.append(int(dets[:, 0].max()))
                if observed:
                    nc = max(nc, max(observed) + 1)
            pred_cls = dets[:, 0].astype(np.float32) if dets.shape[0] else np.zeros((0,), dtype=np.float32)
            pred_conf = dets[:, 1].astype(np.float32) if dets.shape[0] else np.zeros((0,), dtype=np.float32)
            gt_cls = gt[:, 0].astype(np.float32) if gt.shape[0] else np.zeros((0,), dtype=np.float32)
            if gt_cls.size:
                saw_target = True
            if dets.shape[0] == 0 or gt.shape[0] == 0:
                tp = np.zeros((dets.shape[0], IOU_THRESHOLDS.size), dtype=bool)
            else:
                iou_mat = box_iou(torch.from_numpy(gt[:, 1:5]), torch.from_numpy(dets[:, 2:6])).numpy()
                tp = match_predictions(pred_cls, gt_cls, iou_mat)
            metric.update_stats(
                {
                    "tp": tp,
                    "conf": pred_conf,
                    "pred_cls": pred_cls,
                    "target_cls": gt_cls,
                    "target_img": np.unique(gt_cls) if gt_cls.size else np.zeros((0,), dtype=np.float32),
                }
            )
            writer.append(
                {
                    "row": index,
                    "path_img": path_img,
                    "width": orig_w,
                    "height": orig_h,
                    "gt": round_rows(gt, with_score=False),
                    "dets": round_rows(dets, with_score=True),
                }
            )
            ok_count += 1
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"  progress: {index + 1} / {len(rows)} (ok={ok_count} skip={skip_count})")
        if ok_count and saw_target:
            metric.names = {class_id: str(class_id) for class_id in range(max(nc, 1))}
            metric.process(plot=False)
    except BaseException:
        writer.close_partial()
        if writer.staging.is_file():
            print(f"partial dump kept at {writer.staging}", file=sys.stderr)
        raise

    summary = summarize(metric) if ok_count and saw_target else empty_summary()
    summary = {
        "ok": ok_count,
        "skip": skip_count,
        "num_classes": nc,
        **summary,
    }
    writer.finish(summary)
    if ok_count == 0 or summary["map"] is None:
        print("map: n/a")
        print(f"done: {out_path}")
        return 1
    print(f"map50-95: {summary['map']:.6f}  map50: {summary['map50']:.6f}  map75: {summary['map75']:.6f}")
    print(
        f"precision: {summary['precision']:.6f}  recall: {summary['recall']:.6f}"
        f"  at f1_confidence={summary['f1_confidence']}"
    )
    print(f"done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
