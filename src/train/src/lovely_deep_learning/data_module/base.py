import os
from typing import Optional, List
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lovely_deep_learning.dataset.predict import ImagePredictDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_paths: List[str],
        val_csv_paths: List[str],
        test_csv_paths: List[str],
        predict_csv_paths: List[str],
        key_map,
        transform_train,
        transform_val,
        transform_test=None,
        transform_predict=None,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_csv_paths = train_csv_paths
        self.val_csv_paths = val_csv_paths
        self.test_csv_paths = test_csv_paths
        self.predict_csv_paths = predict_csv_paths
        self.key_map = key_map
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test or transform_val
        self.transform_predict = transform_predict or transform_val
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.get_collate_fn_for_dataloader(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.get_collate_fn_for_dataloader(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.get_collate_fn_for_dataloader(),
        )

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
