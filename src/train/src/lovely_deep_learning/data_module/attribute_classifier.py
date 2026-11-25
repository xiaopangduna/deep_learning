# -*- encoding: utf-8 -*-
"""框上的多属性分类。CSV 事先写好，本模块不下载、不生成表。"""

from typing import Optional, Sequence

from torch.utils.data import DataLoader

from .base import BaseDataModule
from ..dataset.attribute_classifier import AttributeFromDetectBoxDataset


class AttributeFromDetectBoxDataModule(BaseDataModule):
    """为各阶段挂载 :class:`AttributeFromDetectBoxDataset`。

    训练、验证、测试共用 ``attribute_columns`` 和 ``key_map``。
    ``predict_attribute_columns`` 为 ``None`` 时沿用 ``attribute_columns``。
    预测表没有属性列时传空列表，只裁图、不读标签。
    ``prepare_data`` 不写 CSV。
    """

    def __init__(
        self,
        attribute_columns: Optional[Sequence[str]] = None,
        predict_attribute_columns: Optional[Sequence[str]] = None,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
        key_map=None,
        predict_key_map=None,
        norm_mean=None,
        norm_std=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        attribute_columns
            训练、验证、测试要读的属性列。``None`` 表示各表里全部 ``attr_`` 列。
        predict_attribute_columns
            预测表的属性列。``None`` 沿用 ``attribute_columns``。空列表表示不读标签。
        box_scale
            裁剪前按中心外扩的倍数。超出图像的部分被截掉。
        min_side_px
            短边小于该像素的框丢掉。
        key_map, predict_key_map
            传给 Dataset。``None`` 使用 Dataset 的默认映射。
        norm_mean, norm_std
            传给 Dataset，供可视化反归一化。``None`` 表示跳过。
        **kwargs
            交给 ``BaseDataModule``（各阶段 CSV、transform、``batch_size``、``num_workers``）。
        """
        super().__init__(**kwargs)
        self.attribute_columns = None if attribute_columns is None else list(attribute_columns)
        self.predict_attribute_columns = (
            None if predict_attribute_columns is None else list(predict_attribute_columns)
        )
        self.box_scale = float(box_scale)
        self.min_side_px = float(min_side_px)
        self.key_map = key_map
        self.predict_key_map = predict_key_map
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def setup(self, stage=None):
        """按 Lightning ``stage`` 创建 Dataset。``stage is None`` 时四个阶段都建。"""
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(
                self.train_csv_paths, self.key_map, self.transform_train, self.attribute_columns
            )
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val, self.attribute_columns
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val, self.attribute_columns
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._make_dataset(
                self.test_csv_paths, self.key_map, self.transform_test, self.attribute_columns
            )
        if stage == "predict" or stage is None:
            columns = (
                self.attribute_columns
                if self.predict_attribute_columns is None
                else self.predict_attribute_columns
            )
            self.pred_dataset = self._make_dataset(
                self.predict_csv_paths,
                self.predict_key_map,
                self.transform_predict,
                columns,
            )

    def _make_dataset(self, csv_paths, key_map, transform, attribute_columns):
        return AttributeFromDetectBoxDataset(
            csv_paths,
            attribute_columns=attribute_columns,
            key_map=key_map,
            transform=transform,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            box_scale=self.box_scale,
            min_side_px=self.min_side_px,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.pred_dataset.get_collate_fn_for_dataloader(),
        )
