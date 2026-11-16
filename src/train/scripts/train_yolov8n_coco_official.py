"""用 Ultralytics 官方 API，按 YOLOv8n COCO 配方从头训练。

结构来自 ``yolov8n.yaml``（随机初始化，不加载 ``yolov8n.pt``）。
超参对齐 Ultralytics 8.3.189 ``cfg/default.yaml``，并显式指定 SGD：
``optimizer=auto`` 会把 momentum 改成 0.9、``warmup_bias_lr`` 改成 0。

默认 500 epoch、``patience=0``、``batch=64``（等于名义 batch ``nbs=64``，
weight decay 保持 5e-4）。验证集 batch 是训练 batch 的 2 倍。

在 src/train 下::

    uv run python scripts/train_yolov8n_coco_official.py
    uv run python scripts/train_yolov8n_coco_official.py --resume
    uv run python scripts/train_yolov8n_coco_official.py --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

from lovely_deep_learning.data_module.coco import COCO80_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = ROOT / "datasets" / "COCO" / "coco"
DATA_YAML = ROOT / "configs" / "datasets" / "coco_local.yaml"
RUNS_DIR = ROOT / "runs" / "ultralytics"


def prepare_data_yaml() -> Path:
    """把 yaml 里的 path 写成绝对路径，避免落到 Ultralytics 默认 datasets 目录。"""
    train_dir = COCO_ROOT / "images" / "train2017"
    val_dir = COCO_ROOT / "images" / "val2017"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"missing COCO train images: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"missing COCO val images: {val_dir}")

    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    cfg = YAML.load(DATA_YAML) if DATA_YAML.is_file() else {}
    cfg["path"] = str(COCO_ROOT.resolve())
    cfg["train"] = "images/train2017"
    cfg["val"] = "images/val2017"
    cfg["names"] = {i: name for i, name in enumerate(COCO80_CLASS_NAMES)}
    YAML.save(DATA_YAML, cfg)
    return DATA_YAML


def run_name(epochs: int, batch: int) -> str:
    return f"coco_official{epochs}_yolov8n_b{batch}"


def train(epochs: int, batch: int, workers: int, patience: int) -> None:
    data = prepare_data_yaml()
    name = run_name(epochs, batch)
    print(f"========== YOLOv8n COCO scratch: {epochs} epochs, batch={batch} ==========")
    print(f"data: {data}")
    print(f"run:  {RUNS_DIR / name}")
    model = YOLO("yolov8n.yaml")
    model.train(
        data=str(data),
        epochs=int(epochs),
        patience=int(patience),
        imgsz=640,
        batch=int(batch),
        workers=int(workers),
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        pretrained=False,
        amp=True,
        seed=0,
        deterministic=True,
        plots=False,
        device=0,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,
    )


def resume(epochs: int, batch: int) -> None:
    last_pt = RUNS_DIR / run_name(epochs, batch) / "weights" / "last.pt"
    if not last_pt.is_file():
        raise FileNotFoundError(f"missing checkpoint to resume: {last_pt}")
    print(f"========== resume from {last_pt} ==========")
    YOLO(str(last_pt)).train(resume=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="训练 batch。验证 batch 为其 2 倍；OOM 时改为 32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="0 关闭早停，跑满 epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从同一 epochs/batch 对应 run 的 weights/last.pt 继续",
    )
    args = parser.parse_args()
    if args.resume:
        resume(args.epochs, args.batch)
    else:
        train(args.epochs, args.batch, args.workers, args.patience)


if __name__ == "__main__":
    main()
