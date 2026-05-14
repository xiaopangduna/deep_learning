from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from lightning.pytorch.cli import instantiate_class


def _load_experiment_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} 须为 YAML 映射。")
    return cfg


def run_export(
    config_path: str | Path,
    ckpt_path: str | Path | None = None,
    export_format: str = "onnx",
) -> str:
    config = _load_experiment_yaml(Path(config_path))
    model_block = config.get("model", {})
    class_path = model_block.get("class_path")
    init_args = dict(model_block.get("init_args", {}))
    if not class_path:
        raise ValueError("配置中缺少 model.class_path。")
    if "model" not in init_args:
        raise ValueError("配置中缺少 model.init_args.model。")

    for key in ("criterion", "postprocess", "metrics"):
        spec = init_args.get(key)
        if isinstance(spec, dict) and "class_path" in spec:
            init_args[key] = instantiate_class((), spec)

    module = instantiate_class(
        (),
        {"class_path": class_path, "init_args": init_args},
    )
    export_fn = getattr(module, "export", None)
    if not callable(export_fn):
        raise ValueError(
            f"{class_path} 实例未提供 export()；请使用继承 BaseModule 的 LightningModule。"
        )
    return export_fn(ckpt_path=ckpt_path, export_format=export_format)


def export_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="通过实验配置导出模型（可选加载 ckpt 权重）。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/object_detect_COCO8.yaml"),
        help="实验配置 YAML（默认 configs/experiments/object_detect_COCO8.yaml）。",
    )
    parser.add_argument("--ckpt_path", type=Path, default=None, help="可选训练 checkpoint 路径；不传则导出当前初始化权重。")
    parser.add_argument("--format", type=str, default="onnx", help="导出格式，支持 onnx / pt。")
    args = parser.parse_args(argv)
    output = run_export(
        config_path=args.config,
        ckpt_path=args.ckpt_path,
        export_format=args.format,
    )
    print(f"导出完成: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="模型导出脚本。")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("export", help="导出模型")

    args, remaining = parser.parse_known_args()
    if args.command == "export":
        export_main(remaining)
        return
    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
