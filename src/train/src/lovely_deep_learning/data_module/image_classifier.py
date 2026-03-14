import os
from typing import Optional, List
from pathlib import Path
import urllib.request
import tarfile
import lightning.pytorch as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .base import BaseDataModule
from lovely_deep_learning.dataset.predict import ImagePredictDataset

from ..dataset.image_classifier import ImageClassifierDataset


class ImageClassifierDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageClassifierDataset(
                self.train_csv_paths,
                key_map=self.key_map,
                transform=self.transform_train,
            )
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ImageClassifierDataset(
                self.test_csv_paths,
                key_map=self.key_map,
                transform=self.transform_test,
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = ImagePredictDataset(self.predict_csv_paths, transform=self.transform_test)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.get_collate_fn_for_dataloader(),
        )

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
