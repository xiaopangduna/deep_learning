"""用官方 Ultralytics API 在本地 COCO 2017 上评估 YOLOv8n。

``val`` 只报 mAP（官方 ``model.val()`` 不算 box/cls/dfl）。
``train`` 跑 1 个 epoch，进度条和 epoch 末 val 会打出官方口径的 loss。

用法（在 src/train 下）::

    python scripts/train_yolov8.py val
    python scripts/train_yolov8.py train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "pretrained_models" / "yolov8n.pt"
COCO_ROOT = ROOT / "datasets" / "COCO" / "coco"
DATA_YAML = ROOT / "configs" / "datasets" / "coco_local.yaml"
RUNS_DIR = ROOT / "runs" / "ultralytics"


def _prepare_data_yaml() -> Path:
    """把 yaml 里的 path 写成绝对路径，避免落到 Ultralytics 默认 datasets 目录。"""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"missing data yaml: {DATA_YAML}")
    val_dir = COCO_ROOT / "images" / "val2017"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"missing COCO val images: {val_dir}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"missing weights: {WEIGHTS}")

    cfg = YAML.load(DATA_YAML)
    cfg["path"] = str(COCO_ROOT.resolve())
    YAML.save(DATA_YAML, cfg)
    return DATA_YAML


def _load_model() -> YOLO:
    model = YOLO(str(WEIGHTS))
    model.info()
    return model


def run_val(model: YOLO, data: Path) -> None:
    """官方 val：letterbox + rect + conf=0.001，只打印 mAP。"""
    metrics = model.val(
        data=str(data),
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,
        plots=False,
        save_json=False,
        project=str(RUNS_DIR),
        name="coco_val_yolov8n",
        exist_ok=True,
    )
    print("========== official val (mAP only, no loss) ==========")
    print(f"mAP50-95: {float(metrics.box.map):.4f}")
    print(f"mAP50:    {float(metrics.box.map50):.4f}")
    print(f"mAP75:    {float(metrics.box.map75):.4f}")
    print(metrics.results_dict)


def run_train_one_epoch(model: YOLO, data: Path) -> None:
    """1 epoch：进度条 box/cls/dfl 为官方日志标度（未乘 batch_size）；epoch 末还有 val loss。"""
    print("========== 1-epoch train (official loss on console) ==========")
    print("COCO train2017 ~118k images; this is much slower than val.")
    model.train(
        data=str(data),
        epochs=1,
        imgsz=640,
        batch=16,
        plots=False,
        project=str(RUNS_DIR),
        name="coco_train1_yolov8n",
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="val",
        choices=("val", "train"),
        help="val=mAP on val2017 (default); train=1 epoch to see official loss",
    )
    args = parser.parse_args()
    data = _prepare_data_yaml()
    model = _load_model()
    if args.mode == "val":
        run_val(model, data)
    else:
        run_train_one_epoch(model, data)


if __name__ == "__main__":
    main()
