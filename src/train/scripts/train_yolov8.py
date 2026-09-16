"""用官方 Ultralytics API 在本地 COCO 2017 上评估 / 微调 / 从零训练 YOLOv8n。

``val`` 只报 mAP（官方 ``model.val()`` 不算 box/cls/dfl）。
``train`` 从 ``yolov8n.pt`` 接着训。
``scratch`` 用 ``yolov8n.yaml`` 随机初始化，按官方 COCO 配方从零训（默认 500 epoch）。

用法（在 src/train 下）::

    python scripts/train_yolov8.py val
    python scripts/train_yolov8.py train --epochs 10
    python scripts/train_yolov8.py train --epochs 10 --lr 1e-3
    python scripts/train_yolov8.py scratch
    python scripts/train_yolov8.py scratch --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "pretrained_models" / "yolov8n.pt"
SCRATCH_CFG = "yolov8n.yaml"
COCO_ROOT = ROOT / "datasets" / "COCO" / "coco"
DATA_YAML = ROOT / "configs" / "datasets" / "coco_local.yaml"
RUNS_DIR = ROOT / "runs" / "ultralytics"


_COCO80_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def _prepare_data_yaml() -> Path:
    """把 yaml 里的 path 写成绝对路径，避免落到 Ultralytics 默认 datasets 目录。"""
    if not DATA_YAML.is_file():
        DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
        YAML.save(
            DATA_YAML,
            {
                "path": str(COCO_ROOT.resolve()),
                "train": "images/train2017",
                "val": "images/val2017",
                "test": "images/test2017",
                "names": _COCO80_NAMES,
            },
        )
    val_dir = COCO_ROOT / "images" / "val2017"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"missing COCO val images: {val_dir}")

    cfg = YAML.load(DATA_YAML)
    cfg["path"] = str(COCO_ROOT.resolve())
    YAML.save(DATA_YAML, cfg)
    return DATA_YAML


def _load_model(*, scratch: bool) -> YOLO:
    if scratch:
        model = YOLO(SCRATCH_CFG)
    else:
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"missing weights: {WEIGHTS}")
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


def run_train(
    model: YOLO,
    data: Path,
    epochs: int,
    lr: float | None,
) -> None:
    """从预训练权重接着训。不传 ``lr`` 时与当前 10 轮实验相同（``optimizer=auto`` → SGD 0.01）。

    传入 ``--lr`` 时固定 SGD，只改学习率（及配套的 ``warmup_bias_lr``），其余与本次
    对照一致：batch=16、nbs=64、close_mosaic=10、workers=2。结果写到独立目录，
    不会覆盖 ``coco_train{epochs}_yolov8n``。
    """
    kwargs: dict = {
        "data": str(data),
        "epochs": int(epochs),
        "imgsz": 640,
        "batch": 16,
        "workers": 2,
        "plots": False,
        "project": str(RUNS_DIR),
        "exist_ok": True,
    }
    if lr is None:
        kwargs["name"] = f"coco_train{epochs}_yolov8n"
        print(f"========== {epochs}-epoch train (official auto / SGD 0.01) ==========")
    else:
        # auto 会忽略 lr0；必须显式 SGD，才能只改学习率。
        # warmup_bias_lr 默认 0.1，不跟着降的话前几步仍会打散预训练权重。
        kwargs["optimizer"] = "SGD"
        kwargs["lr0"] = float(lr)
        kwargs["warmup_bias_lr"] = float(lr)
        kwargs["name"] = f"coco_train{epochs}_yolov8n_sgd_lr{lr:g}"
        print(
            f"========== {epochs}-epoch train (SGD lr0={lr:g}, warmup_bias_lr={lr:g}) =========="
        )
    print("COCO train2017 ~118k images; this is much slower than val.")
    model.train(**kwargs)


def run_scratch(model: YOLO | None, data: Path, epochs: int, *, resume: bool = False) -> None:
    """从 ``yolov8n.yaml`` 随机初始化，按官方 COCO 从头训配方。

    官方复现常用 4 卡 ``batch=128``、500 epoch、SGD ``lr0=0.01``、mosaic，最后 10 轮
    ``close_mosaic``。本机 4060 8GB / WSL ~8GB RAM 改为 ``batch=16``（``nbs=64``
    仍按 64 等效积累）、``workers=2``。BN 统计和有效 batch 与 128 不完全一致，
    终点 mAP 可能略低于公布的 ~0.373。

    ``resume=True`` 时加载同名 run 的 ``weights/last.pt``，恢复 epoch / 优化器 / 学习率。
    """
    run_name = f"coco_scratch{epochs}_yolov8n"
    last_pt = RUNS_DIR / run_name / "weights" / "last.pt"
    if resume:
        if not last_pt.is_file():
            raise FileNotFoundError(f"missing checkpoint to resume: {last_pt}")
        print(f"========== resume scratch from {last_pt} ==========")
        YOLO(str(last_pt)).train(resume=True)
        return

    hours = epochs * 18 / 60
    print(f"========== {epochs}-epoch scratch (yolov8n.yaml, SGD 0.01) ==========")
    print(
        f"COCO train2017 ~118k images; ~17–18 min/epoch on 4060 → roughly {hours:.0f} h "
        f"for {epochs} epochs. Official paper used batch=128; this run uses batch=16."
    )
    model.train(
        data=str(data),
        epochs=int(epochs),
        imgsz=640,
        batch=16,
        workers=2,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        mosaic=1.0,
        close_mosaic=10,
        pretrained=False,
        amp=True,
        plots=False,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="val",
        choices=("val", "train", "scratch"),
        help="val=mAP; train=finetune yolov8n.pt; scratch=from yolov8n.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch 数。train 默认 10；scratch 默认 500（官方 COCO）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="仅 train：显式 SGD 学习率。省略则 optimizer=auto（SGD 0.01）；"
        "低 lr 对照建议 1e-3",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="scratch：从 runs/ultralytics/coco_scratch{epochs}_yolov8n/weights/last.pt 继续",
    )
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 500 if args.mode == "scratch" else 10
    data = _prepare_data_yaml()
    if args.mode == "scratch" and args.resume:
        run_scratch(None, data, args.epochs, resume=True)
        return
    model = _load_model(scratch=args.mode == "scratch")
    if args.mode == "val":
        run_val(model, data)
    elif args.mode == "scratch":
        run_scratch(model, data, args.epochs)
    else:
        run_train(model, data, args.epochs, args.lr)


if __name__ == "__main__":
    main()
