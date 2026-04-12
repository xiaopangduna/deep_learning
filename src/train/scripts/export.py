"""从 Lightning checkpoint 导出 ONNX（与 ``ObjectDetectModule`` / DAGNet(YOLOv8) 训练配置一致）。

在 ``src/train`` 目录下执行::

    PYTHONPATH=src python scripts/export.py --config configs/experiments/object_detect_COCO8_export.yaml

可选：``--ckpt`` / ``--output`` 覆盖 YAML 中的路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TRAIN_ROOT = Path(__file__).resolve().parents[1]
_SRC = _TRAIN_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import yaml

from lovely_deep_learning.module.object_detect import ObjectDetectModule


def _load_export_block(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "export" not in cfg:
        raise ValueError(f"{config_path} 须包含 ``export:`` 段（ckpt_path、output_path 等）。")
    exp = cfg["export"]
    if not isinstance(exp, dict):
        raise ValueError(f"{config_path} 中 ``export`` 须为映射。")
    return exp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 object_detect 等实验的 Lightning ckpt 导出为 ONNX。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/object_detect_COCO8_export.yaml"),
        help="含 ``export:`` 的 YAML（默认 COCO8 导出示例）。",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="覆盖 YAML 中的 ``export.ckpt_path``。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="覆盖 YAML 中的 ``export.output_path``。",
    )
    args = parser.parse_args()

    exp = _load_export_block(args.config)
    fmt = str(exp.get("format", "onnx")).lower()
    if fmt != "onnx":
        raise ValueError(f"当前仅支持 format=onnx，收到: {fmt!r}")

    ckpt_path = Path(args.ckpt) if args.ckpt is not None else Path(exp["ckpt_path"])
    output_path = Path(args.output) if args.output is not None else Path(exp["output_path"])
    opset = int(exp.get("opset", 17))
    input_shape = exp.get("input_shape", [1, 3, 640, 640])
    if len(input_shape) != 4:
        raise ValueError(f"input_shape 须为 [N,C,H,W]，当前: {input_shape}")

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"未找到 checkpoint: {ckpt_path.resolve()}（请在 src/train 下执行，或检查路径）"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载 checkpoint: {ckpt_path}")
    module = ObjectDetectModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
    )
    module.eval()

    n, c, h, w = (int(x) for x in input_shape)
    dummy = torch.randn(n, c, h, w)

    print(f"导出 ONNX → {output_path}（opset={opset}, 输入形状={input_shape}）")
    module.to_onnx(
        output_path,
        dummy,
        export_params=True,
        opset_version=opset,
        input_names=["images"],
    )
    print("导出完成。")


if __name__ == "__main__":
    main()
