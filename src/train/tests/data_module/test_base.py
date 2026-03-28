from torch.utils.data import Dataset
from lovely_deep_learning.data_module.base import BaseDataModule


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"x": idx}

    def get_collate_fn_for_dataloader(self):
        return lambda batch: batch

def test_BaseDataModule_init():
    data_module = BaseDataModule(
        train_csv_paths=["tests/test_data/dataset/test_image_classifier_train.csv"],
        val_csv_paths=["tests/test_data/dataset/test_image_classifier_train.csv"],
        test_csv_paths=["tests/test_data/dataset/test_image_classifier_train.csv"],
        predict_csv_paths=["tests/test_data/dataset/test_image_classifier_predict.csv"],
        transform_train=None,
        transform_val=None,
        transform_test=None,
        transform_predict=None,
    )
    data_module.train_dataset = DummyDataset()
    data_module.val_dataset = DummyDataset()
    data_module.test_dataset = DummyDataset()
    data_module.pred_dataset = DummyDataset()
    assert data_module.train_dataloader() is not None
    assert data_module.val_dataloader() is not None
    assert data_module.test_dataloader() is not None
    assert data_module.predict_dataloader() is not None