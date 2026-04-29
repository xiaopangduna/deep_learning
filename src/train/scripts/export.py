from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_TRAIN_ROOT = Path(__file__).resolve().parents[1]
_SRC = _TRAIN_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import yaml
from lightning.pytorch.cli import instantiate_class

from lovely_deep_learning.model.DAGNet import DAGNet


def _load_experiment_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} 须为 YAML 映射。")
    return cfg


def run_export(
    config_path: str | Path,
    ckpt_path: str | Path | None = None,
    format: str = "onnx",
) -> str:
    config = _load_experiment_yaml(Path(config_path))
    model_block = config.get("model", {})
    class_path = model_block.get("class_path")
    init_args = dict(model_block.get("init_args", {}))
    if not class_path:
        raise ValueError("配置中缺少 model.class_path。")
    if "model" not in init_args:
        raise ValueError("配置中缺少 model.init_args.model。")

    exporter_cfg = init_args["model"].get("exporter", {}) or {}
    fmt = str(format).lower()
    suffix_by_format = {"onnx": ".onnx", "pt": ".pt", "engine": ".engine", "trt": ".engine", "tensorrt": ".engine"}
    default_suffix = suffix_by_format.get(fmt, f".{fmt}")
    if ckpt_path is not None:
        ckpt = Path(ckpt_path).resolve()
        default_output = ckpt.parent / f"{ckpt.stem}{default_suffix}"
    else:
        model_name = init_args["model"].get("model_name", "undefined")
        default_output = Path.cwd() / f"{model_name}{default_suffix}"

    if isinstance(exporter_cfg, dict) and "class_path" in exporter_cfg:
        exporter_spec = dict(exporter_cfg)
        exporter_init_args = dict(exporter_spec.get("init_args") or {})
        if fmt == "pt":
            pt_cfg = dict(exporter_init_args.get("pt_cfg") or {})
            pt_cfg.setdefault("output_path", str(default_output))
            exporter_init_args["pt_cfg"] = pt_cfg
        else:
            onnx_cfg = dict(exporter_init_args.get("onnx_cfg") or {})
            onnx_cfg.setdefault("output_path", str(default_output))
            exporter_init_args["onnx_cfg"] = onnx_cfg
        exporter_spec["init_args"] = exporter_init_args
        init_args["model"]["exporter"] = exporter_spec
    else:
        if fmt == "pt":
            pt_cfg = dict(exporter_cfg.get("pt_cfg") or {})
            pt_cfg.setdefault("output_path", str(default_output))
            exporter_cfg["pt_cfg"] = pt_cfg
        else:
            onnx_cfg = dict(exporter_cfg.get("onnx_cfg") or {})
            onnx_cfg.setdefault("output_path", str(default_output))
            exporter_cfg["onnx_cfg"] = onnx_cfg
        init_args["model"]["exporter"] = exporter_cfg

    for key in ("criterion", "postprocess", "metrics"):
        spec = init_args.get(key)
        if isinstance(spec, dict) and "class_path" in spec:
            init_args[key] = instantiate_class((), spec)

    module = instantiate_class(
        (),
        {"class_path": class_path, "init_args": init_args},
    )
    if not hasattr(module, "model") or not isinstance(module.model, DAGNet):
        raise ValueError(f"{class_path} 未暴露 DAGNet 到 `model` 属性，无法导出。")
    dagnet = module.model

    if ckpt_path is not None:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", {})
        if not state_dict:
            raise ValueError(f"checkpoint 中缺少 `state_dict`: {ckpt_path}")
        model_state = {
            key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")
        }
        if not model_state:
            raise ValueError("checkpoint 中未找到 `model.` 前缀权重。")
        dagnet.load_state_dict(model_state, strict=False)

    return dagnet.export(format=format)


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
        format=args.format,
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
