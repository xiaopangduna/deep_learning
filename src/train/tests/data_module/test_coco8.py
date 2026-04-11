"""验证 ``COCO8DataModule.prepare_data``：使用项目内 ``datasets/COCO8``，已存在数据则跳过下载。"""

from pathlib import Path
import pytest
import pandas as pd
from torchvision.transforms import v2
import torch

from lovely_deep_learning.data_module.coco import COCO8DataModule, COCO80_CLASS_NAMES

# 与 ``COCO8DataModule`` 默认一致：``src/train/datasets/COCO8`` 下放 CSV，``coco8/`` 为图像与标签
_COCO8_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "COCO8"

KEY_MAP = {"img_path": "path_img", "object_label_path": "path_label_detect_yolo"}
PREDICT_KEY_MAP = {"img_path": "path_img"}
TRANSFORM = v2.Compose(
    [v2.Resize(size=(640, 640)), v2.ToDtype(dtype=torch.float32, scale=True)]
)

@pytest.mark.download
def test_COCO8DataModule_prepare_data():
    """在 ``datasets/COCO8`` 上调用 ``prepare_data``：无数据则下载解压，已有 ``coco8/images/...`` 则跳过下载。"""
    dataset_dir = _COCO8_ROOT / "coco8"

    dm = COCO8DataModule(
        train_csv_paths=[str(_COCO8_ROOT / "train.csv")],
        val_csv_paths=[str(_COCO8_ROOT / "val.csv")],
        test_csv_paths=[str(_COCO8_ROOT / "val.csv")],
        predict_csv_paths=[str(_COCO8_ROOT / "predict.csv")],
        transform_train=TRANSFORM,
        transform_val=TRANSFORM,
        transform_test=TRANSFORM,
        transform_predict=TRANSFORM,
        batch_size=1,
        num_workers=0,
        key_map=KEY_MAP,
        predict_key_map=PREDICT_KEY_MAP,
        map_class_id_to_class_name=None,
        dataset_dir=str(dataset_dir),
    )
    dm.prepare_data()

    assert (dataset_dir / "images" / "train").is_dir()
    assert (dataset_dir / "images" / "val").is_dir()
    assert (dataset_dir / "labels" / "train").is_dir()
    assert (dataset_dir / "labels" / "val").is_dir()

    assert (_COCO8_ROOT / "train.csv").is_file()
    assert (_COCO8_ROOT / "val.csv").is_file()
    assert (_COCO8_ROOT / "predict.csv").is_file()
    assert (_COCO8_ROOT / "map_class_id_to_class_name.csv").is_file()

    train_df = pd.read_csv(_COCO8_ROOT / "train.csv")
    assert list(train_df.columns) == ["path_img", "path_label_detect_yolo"]
    assert len(train_df) == 4

    pred_df = pd.read_csv(_COCO8_ROOT / "predict.csv")
    assert list(pred_df.columns) == ["path_img"]
    assert len(pred_df) == 4

    map_df = pd.read_csv(_COCO8_ROOT / "map_class_id_to_class_name.csv")
    assert len(map_df) == len(COCO80_CLASS_NAMES)
