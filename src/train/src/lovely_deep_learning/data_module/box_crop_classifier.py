"""框级分类 DataModule。转换由脚本完成，这里只把 CSV 交给 ``BoxCropClassifierDataset``。"""

from .image_classifier import ImageClassifierDataModule
from ..dataset.box_crop_classifier import BoxCropClassifierDataset


class BoxCropClassifierDataModule(ImageClassifierDataModule):
    """与图像分类 DataModule 的 CSV、transform、``key_map`` 相同，额外传入 ``box_scale``。

    模型、损失、指标、后处理仍用 ``ImageClassifierModule`` 那一套。
    ``prepare_data`` 不生成 CSV，先运行 ``scripts/yolo_to_box_crop_csv.py``。
    """

    def __init__(self, box_scale: float = 1.2, min_side_px: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.box_scale = float(box_scale)
        self.min_side_px = float(min_side_px)

    def _make_dataset(self, csv_paths, key_map, transform):
        return BoxCropClassifierDataset(
            csv_paths,
            key_map=key_map,
            transform=transform,
            map_class_id_to_class_name=self.map_class_id_to_class_name,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            box_scale=self.box_scale,
            min_side_px=self.min_side_px,
        )

    def setup(self, stage=None):
        self.map_class_id_to_class_name = self._resolve_map_class_id_spec(
            self._map_class_id_to_class_name_spec
        )
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(
                self.train_csv_paths, self.key_map, self.transform_train
            )
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._make_dataset(
                self.test_csv_paths, self.key_map, self.transform_test
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = self._make_dataset(
                self.predict_csv_paths, self.predict_key_map, self.transform_predict
            )
