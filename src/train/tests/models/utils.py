"""从 ``configs/models/*.yaml`` 加载 DAG 模型配置，供 ``test_DAGNet`` 等使用。"""
from pathlib import Path

import yaml

# tests/models/utils.py -> 仓库根为 src/train（含 configs/）
_MODEL_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "models"


def load_model_config(stem: str) -> dict:
    """读取 ``configs/models/{stem}.yaml`` 并解析为字典。"""
    path = _MODEL_CONFIG_DIR / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty or invalid YAML: {path}")
    return data


demo_config_two_inputs_one_outputs = load_model_config("demo_two_inputs_one_outputs")
resnet18_config = load_model_config("resnet18")
efficientnet_v2_s_config = load_model_config("efficientnet_v2_s")
swin_v2_t_config = load_model_config("swin_v2_t")
regnet_y_32gf_config = load_model_config("regnet_y_32gf")
mobilenet_v3_large_config = load_model_config("mobilenet_v3_large")
yolov8_n_config = load_model_config("yolov8_n")
