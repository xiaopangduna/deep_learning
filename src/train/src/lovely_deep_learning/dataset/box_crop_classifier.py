# -*- encoding: utf-8 -*-
"""按框级分类 CSV 从原图裁出单个目标，返回值与图像分类数据集一致。"""

from typing import Any, Callable, Dict, Optional, Sequence, Union
from pathlib import Path

import numpy as np
from torchvision import tv_tensors

from .base import BaseDataset
from .image_classifier import ImageClassifierDataset


def crop_cxcywh(
    img: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    box_scale: float,
) -> np.ndarray:
    """归一化 ``cxcywh`` 外扩 ``box_scale`` 后裁切，结果至少 1 像素。``img`` 为 ``(H, W, C)``。"""
    height, width = img.shape[:2]
    abs_w = max(w * width * box_scale, 1.0)
    abs_h = max(h * height * box_scale, 1.0)
    abs_cx = cx * width
    abs_cy = cy * height
    x1 = int(np.floor(abs_cx - abs_w / 2.0))
    y1 = int(np.floor(abs_cy - abs_h / 2.0))
    x2 = int(np.ceil(abs_cx + abs_w / 2.0))
    y2 = int(np.ceil(abs_cy + abs_h / 2.0))
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return img[y1:y2, x1:x2]


class BoxCropClassifierDataset(ImageClassifierDataset):
    """一行一个框。CSV 里的 ``class_id`` 是 COCO 原始 id。

    初始化时筛选：丢掉 ``w`` 或 ``h`` 非正的框；短边 ``min(w*img_w, h*img_h)`` 小于
    ``min_side_px`` 的框；有标签时只保留 ``map_class_id_to_class_name`` 里的类别名，
    并把 ``class_id`` 改写成该映射的连续 id。``predict`` 可以没有类别列。
    ``box_scale`` 默认 1.2，裁剪前按中心外扩。缩放交给外部 ``transform``。
    """

    DEFAULT_KEY_MAP = {
        "img_path": "path_img",
        "class_name": "class_name",
        "class_id": "class_id",
        "cx": "cx",
        "cy": "cy",
        "w": "w",
        "h": "h",
        "img_w": "img_w",
        "img_h": "img_h",
    }
    PREDICT_KEY_MAP = {
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
        key_map: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
        norm_mean: Optional[list[float]] = None,
        norm_std: Optional[list[float]] = None,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
    ):
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
        if norm_mean is None:
            norm_mean = [0.485, 0.456, 0.406]
        if norm_std is None:
            norm_std = [0.229, 0.224, 0.225]
        target_map = self._normalize_map_class_id_to_class_name(map_class_id_to_class_name)
        super().__init__(
            csv_paths=csv_paths,
            key_map=key_map,
            transform=transform,
            map_class_id_to_class_name=None,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
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
        self._filter_boxes(target_map)

    def _filter_boxes(self, target_map: Dict[int, str]) -> None:
        """按几何条件和类别名子集改写 ``sample_path_table``，并重映射 ``class_id``。"""
        table = self.sample_path_table
        width = table["w"].astype(float)
        height = table["h"].astype(float)
        keep = (width > 0) & (height > 0)
        if self.min_side_px > 0:
            short_side = np.minimum(
                width * table["img_w"].astype(float),
                height * table["img_h"].astype(float),
            )
            keep = keep & (short_side >= self.min_side_px)
        if self._has_label and target_map:
            names = set(target_map.values())
            keep = keep & table["class_name"].isin(names)
        filtered = table.loc[keep].reset_index(drop=True)
        if self._has_label and target_map:
            name_to_id = {name: class_id for class_id, name in target_map.items()}
            filtered["class_id"] = filtered["class_name"].map(name_to_id)
        self.sample_path_table = filtered
        self.num_samples = len(filtered)
        self.map_class_id_to_class_name = target_map
        if self._has_label and target_map:
            if self.num_samples == 0:
                raise ValueError(
                    "筛选后没有样本。检查 map_class_id_to_class_name 与 min_side_px"
                )
            self._validate_class_mapping()

    def __getitem__(self, index):
        row = self.sample_path_table.iloc[index]
        img_path = str(row["img_path"])
        img_np, img_shape = self.read_img(img_path, None)
        crop = crop_cxcywh(
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
        net_out: Dict[str, Any] = {}
        if self._has_label:
            net_out["class_name"] = row["class_name"]
            net_out["class_id"] = int(row["class_id"])
        return net_in, net_out
