import pytest
from pathlib import Path

from lovely_deep_learning.datamodules.COCO8_data_module import COCO8DataModule


@pytest.mark.download
def test_prepare_data():
    """
    测试 COCO8 数据集能否正常下载。
    仅验证下载和解压过程是否顺利，不验证文件内容。
    """
    data_dir = Path("./datasets")  # 使用默认路径
    datamodule = COCO8DataModule(data_dir=str(data_dir), download=True)

    try:
        datamodule.prepare_data()
    except Exception as e:
        pytest.fail(f"COCO8 数据集下载失败: {e}")

    # 简单断言目录存在
    train_path = data_dir /"coco8"/ "images" / "train"
    val_path = data_dir / "coco8" / "images" / "val"
    assert train_path.exists(), "train 目录不存在，下载可能失败"
    assert val_path.exists(), "val 目录不存在，下载可能失败"


def test_setup():
    pass
