# -*- encoding: utf-8 -*-
"""一行一个框，多列 ``attr_`` 标签。

``net_out["attr"]`` 是 ``float32`` 向量，顺序与 :attr:`attribute_columns` 一致。
没有属性列时 ``net_out`` 为空字典。裁剪使用框级分类的 ``crop_cxcywh``：
按 ``box_scale`` 外扩，再截回图像内。
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import pandas as pd
import torch
from pathlib import Path
from torchvision import tv_tensors

from .base import BaseDataset
from .box_crop_distribution import box_keep_masks
from .image_classifier import ImageClassifierDataset, ImageClassifierFromDetectBoxDataset


class AttributeFromDetectBoxDataset(BaseDataset):
    """按框裁图，并把选中的属性列收成一个 0/1 向量。

    ``key_map`` 只映射图像路径和框。属性列保留 CSV 表头，不要写进 ``key_map``。
    ``attribute_columns`` 为 ``None`` 时，按表头顺序收集全部 ``attr_`` 列。
    传入名单时，这些列必须都在表头里，顺序就是向量顺序。空名单表示不读标签。

    丢掉 ``w`` 或 ``h`` 非正的框，以及短边 ``min(w*img_w, h*img_h)`` 小于
    ``min_side_px`` 的框。属性值在筛选后检查，只能是 ``0`` 或 ``1``。
    """

    ATTR_PREFIX = "attr_"
    DEFAULT_KEY_MAP = {
        "img_path": "path_img",
        "cx": "cx",
        "cy": "cy",
        "w": "w",
        "h": "h",
        "img_w": "img_w",
        "img_h": "img_h",
    }

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        attribute_columns: Optional[Sequence[str]] = None,
        key_map: Optional[dict[str, str]] = None,
        transform: Optional[Callable] = None,
        norm_mean: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_MEAN,
        norm_std: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_STD,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
    ):
        """
        Parameters
        ----------
        csv_paths
            至少一张已存在的 CSV。约定同 ``BaseDataset``。
        attribute_columns
            要读的属性列。``None`` 表示表头里全部 ``attr_`` 列。空名单表示没有标签。
        key_map
            ``None`` 复制 :attr:`DEFAULT_KEY_MAP`。
        transform
            作用于裁剪后 ``tv_tensors.Image`` 的变换。``None`` 表示不变换。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，留给可视化反归一化。
            省略时使用 ImageNet 常数。``None`` 表示跳过均值方差反归一化。
        box_scale
            裁剪前按中心外扩的倍数，须为正数。外扩超出图像的部分被截掉。
        min_side_px
            短边小于该像素的框丢掉。``0`` 表示不按短边筛选。
        """
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.norm_mean = None if norm_mean is None else list(norm_mean)
        self.norm_std = None if norm_std is None else list(norm_std)
        missing = [
            name
            for name in ("cx", "cy", "w", "h", "img_w", "img_h")
            if name not in self.sample_path_table.columns
        ]
        if missing:
            raise ValueError(f"框级 CSV 缺少列 {missing}")
        if box_scale <= 0:
            raise ValueError(f"box_scale 须为正数，收到 {box_scale}")
        if min_side_px < 0:
            raise ValueError(f"min_side_px 不能为负，收到 {min_side_px}")
        self.box_scale = float(box_scale)
        self.min_side_px = float(min_side_px)
        self.attribute_columns = self.resolve_attribute_columns(
            list(self.sample_path_table.columns),
            attribute_columns,
        )
        self._filter_boxes()
        self._validate_attr_values()

    @staticmethod
    def resolve_attribute_columns(
        fieldnames: Sequence[str],
        attribute_columns: Optional[Sequence[str]],
    ) -> list[str]:
        """返回本次使用的列名。名单里的列不在表头、或名单自身重复时抛出 ``ValueError``。"""
        if attribute_columns is None:
            return [name for name in fieldnames if str(name).startswith(AttributeFromDetectBoxDataset.ATTR_PREFIX)]
        columns = [str(name) for name in attribute_columns]
        if len(columns) != len(set(columns)):
            raise ValueError(f"attribute_columns 有重复列名：{columns}")
        missing = [name for name in columns if name not in fieldnames]
        if missing:
            raise ValueError(f"attribute_columns 不在 CSV 表头中：{missing}")
        return columns

    @staticmethod
    def parse_attr_values(row: pd.Series, columns: Sequence[str]) -> torch.Tensor:
        """读出 ``(K,)`` ``float32``。每个值只能是 ``0`` 或 ``1``。"""
        values = []
        for column in columns:
            text = str(row[column]).strip()
            if text == "0":
                values.append(0.0)
            elif text == "1":
                values.append(1.0)
            else:
                raise ValueError(f"属性列 {column} 只能是 0 或 1，收到 {text!r}")
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index: int):
        row = self.sample_path_table.iloc[index]
        img_path = str(row["img_path"])
        img_np, img_shape = self.read_img(img_path, None)
        crop = ImageClassifierFromDetectBoxDataset.crop_cxcywh(
            img_np,
            float(row["cx"]),
            float(row["cy"]),
            float(row["w"]),
            float(row["h"]),
            self.box_scale,
        )
        img_tv = tv_tensors.Image(BaseDataset.convert_img_from_numpy_to_tensor_uint8(crop))
        if self.transform:
            img_tv = self.transform(img_tv)
        net_in = {
            "img_path": img_path,
            "img_shape": img_shape,
            "img_tv_transformed": img_tv,
        }
        if not self.attribute_columns:
            return net_in, {}
        return net_in, {"attr": self.parse_attr_values(row, self.attribute_columns)}

    @staticmethod
    def get_collate_fn_for_dataloader():
        """把 ``attr`` 堆成 ``(B, K)`` ``float32``。整批都没有标签时 ``net_out`` 为 ``{}``。"""

        def collate_fn(samples):
            net_in = BaseDataset.collate_net_in_tuple(samples)
            net_outs = [net_out for _, net_out in samples]
            if not net_outs or all(not net_out for net_out in net_outs):
                return net_in, {}
            if any("attr" not in net_out for net_out in net_outs):
                raise ValueError("同一批样本必须同时有或没有属性标签")
            return net_in, {"attr": torch.stack([net_out["attr"] for net_out in net_outs], dim=0)}

        return collate_fn

    def _filter_boxes(self) -> None:
        table = self.sample_path_table
        masks = box_keep_masks(
            table,
            min_side_px=self.min_side_px,
            has_label=False,
            class_groups=None,
        )
        filtered = table.loc[masks["keep"]].reset_index(drop=True)
        if len(filtered) == 0:
            raise ValueError("筛选后没有样本。检查 min_side_px")
        self.sample_path_table = filtered

    def _validate_attr_values(self) -> None:
        table = self.sample_path_table
        for column in self.attribute_columns:
            bad = ~table[column].astype(str).str.strip().isin({"0", "1"})
            if bool(bad.any()):
                shown = sorted(set(table.loc[bad, column].astype(str)))[:8]
                raise ValueError(f"属性列 {column} 只能是 0 或 1，收到 {shown}")
