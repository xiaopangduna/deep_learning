import pytest
from pathlib import Path

from lovely_deep_learning.data_module.image_nette import ImageNetteDataModule


class TestImageNetteDataModule:

    def create_datamodule(self):
        return ImageNetteDataModule(
            data_dir="./datasets/IMAGENETTE/imagenette2-320",
            batch_size=8,
            num_workers=0,
        )

    @pytest.mark.download
    def test_prepare_data(self):
        dm = self.create_datamodule()
        dm.prepare_data()

    def test_setup(self):
        dm = self.create_datamodule()
        dm.setup(stage="fit")

        assert dm.train_dataset is not None, "train_dataset 应该被初始化"
        assert dm.val_dataset is not None, "val_dataset 应该被初始化"

    def test_train_dataloader(self):
        dm = self.create_datamodule()
        dm.setup(stage="fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        x, y = batch

        assert x.shape[0] == dm.batch_size, "训练 batch 大小应等于 batch_size"
        assert x.shape[1:] == (3, 224, 224), "训练图像尺寸应为 (3,224,224)"
        assert y.shape[0] == dm.batch_size, "标签数量应与 batch_size 一致"

    def test_val_dataloader(self):
        dm = self.create_datamodule()
        dm.setup(stage="fit")

        loader = dm.val_dataloader()
        batch = next(iter(loader))
        x, y = batch

        assert x.shape[0] == dm.batch_size, "验证 batch 大小应等于 batch_size"
        assert x.shape[1:] == (3, 224, 224), "验证图像尺寸应为 (3,224,224)"
        assert y.shape[0] == dm.batch_size, "验证标签数量应与 batch_size 一致"
