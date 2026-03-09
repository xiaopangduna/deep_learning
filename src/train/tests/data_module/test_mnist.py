import pytest
from pathlib import Path

from lovely_deep_learning.data_module.mnist import MNISTDataModule


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets"/ "MNIST"


class TestMNISTDataModule:

    def create_datamodule(self):
        return MNISTDataModule(
            data_dir=DATA_DIR,
            batch_size=32,
            num_workers=0,
        )

    # @pytest.mark.download
    # def test_prepare_data(self):
    #     dm = self.create_datamodule()
    #     dm.prepare_data()

    def test_setup(self):
        dm = self.create_datamodule()
        dm.setup("fit")

        assert dm.mnist_train is not None
        assert dm.mnist_val is not None

    def test_train_dataloader(self):
        dm = self.create_datamodule()
        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        x, y = batch
        assert x.shape[0] == dm.batch_size

    def test_val_dataloader(self):
        dm = self.create_datamodule()
        dm.setup("fit")

        loader = dm.val_dataloader()
        batch = next(iter(loader))

        x, y = batch
        assert x.shape[0] == dm.batch_size
