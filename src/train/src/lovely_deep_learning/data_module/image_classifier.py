from .base import BaseDataModule
from ..dataset.image_classifier import ImageClassifierDataset


class ImageClassifierDataModule(BaseDataModule):
    def __init__(self, map_class_id_to_class_name={}, norm_mean=None, norm_std=None, **kwargs):
        super().__init__(**kwargs)
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageClassifierDataset(
                self.train_csv_paths,
                key_map=self.key_map,
                transform=self.transform_train,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std
            )
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std
            )
        if stage == "validate" or stage is None:
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std
            )
        if stage == "test" or stage is None:
            self.test_dataset = ImageClassifierDataset(
                self.test_csv_paths,
                key_map=self.key_map,
                transform=self.transform_test,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = ImageClassifierDataset(
                self.predict_csv_paths,
                transform=self.transform_predict,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std
            )