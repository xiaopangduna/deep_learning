# -*- encoding: utf-8 -*-
"""按框级分类 CSV 从原图裁出单个目标，返回值与图像分类数据集一致。"""

from typing import Any, Callable, Dict, Optional, Sequence, Union
from pathlib import Path

import numpy as np
from torchvision import tv_tensors

from .base import BaseDataset
from .box_crop_distribution import assemble_distribution, box_keep_masks
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


def normalize_class_groups(
    class_groups: Optional[Dict[Any, Sequence[int]]],
) -> Optional[Dict[int, list[int]]]:
    """把 ``{训练 id: [原始 class_id, ...]}`` 规范成键为 ``0..n-1`` 的分组。

    ``None`` 或空字典表示不合并。原始 id 不能重复出现，组不能为空。
    """
    if not class_groups:
        return None
    groups: Dict[int, list[int]] = {}
    for key, members in class_groups.items():
        group_id = int(key)
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise TypeError(
                f"class_groups[{group_id}] 须为原始 class_id 列表，收到 {members!r}"
            )
        original_ids = [int(member) for member in members]
        if not original_ids:
            raise ValueError(f"class_groups[{group_id}] 为空")
        if len(original_ids) != len(set(original_ids)):
            raise ValueError(f"class_groups[{group_id}] 含重复 id: {original_ids}")
        groups[group_id] = original_ids
    expected = set(range(len(groups)))
    if set(groups) != expected:
        last = len(groups) - 1
        raise ValueError(
            f"class_groups 的键须为连续的 0..{last}，收到 {sorted(groups)}"
        )
    seen: Dict[int, int] = {}
    for group_id, original_ids in groups.items():
        for original_id in original_ids:
            if original_id in seen:
                raise ValueError(
                    f"原始 class_id {original_id} 同时出现在组 {seen[original_id]} 和组 {group_id}"
                )
            seen[original_id] = group_id
    return groups


class BoxCropClassifierDataset(ImageClassifierDataset):
    """一行一个框。CSV 里的 ``class_id`` 是原始 id。

    初始化时筛选：丢掉 ``w`` 或 ``h`` 非正的框；短边 ``min(w*img_w, h*img_h)`` 小于
    ``min_side_px`` 的框。有标签时：

    - 传入 ``class_groups`` 则按组合并。键是合并后的训练 id（须为 ``0..n-1``），
      值是原始 ``class_id``。未出现在任何组里的原始类丢掉。样本的 ``class_id`` 改成组号，
      ``class_name`` 改成组内原始类名用 ``+`` 拼接，并写入 ``map_class_id_to_class_name``。
    - 未传 ``class_groups`` 时，只保留 ``map_class_id_to_class_name`` 里的类别名，
      并把 ``class_id`` 改写成该映射的 id。

    ``predict`` 可以没有类别列。``box_scale`` 默认 1.2，裁剪前按中心外扩。缩放交给外部 ``transform``。

    筛选结果写在 ``distribution``（计数、类别、几何）。``kept_image_paths`` 是保留框所属图像，
    供 DataModule 算 train/val/test 的路径交集。这里不打印，由 DataModule 在 ``fit`` 时汇总。
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
        class_groups: Optional[Dict[Any, Sequence[int]]] = None,
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
            validate_class_mapping=False,
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
        self.class_groups = normalize_class_groups(class_groups)
        self._filter_boxes(target_map)

    @staticmethod
    def _names_by_original_id(table) -> Dict[int, str]:
        found: Dict[int, str] = {}
        for raw_id, raw_name in zip(table["class_id"], table["class_name"]):
            text = str(raw_id).strip()
            if text == "":
                continue
            original_id = int(text)
            name = str(raw_name).strip()
            previous = found.get(original_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"原始 class_id {original_id} 对应多个 class_name：{previous!r} 与 {name!r}"
                )
            found[original_id] = name
        return found

    def _merged_class_map(self, table) -> Dict[int, str]:
        names_by_id = self._names_by_original_id(table)
        missing = sorted(
            {
                original_id
                for members in self.class_groups.values()
                for original_id in members
                if original_id not in names_by_id
            }
        )
        if missing:
            raise ValueError(
                f"class_groups 中的原始 class_id 在 CSV 里不存在: {missing}"
            )
        return {
            group_id: "+".join(names_by_id[original_id] for original_id in members)
            for group_id, members in self.class_groups.items()
        }

    def _member_counts(self, table, keep) -> Dict[int, list]:
        kept = table.loc[keep]
        original_ids = kept["class_id"].map(
            lambda value: int(str(value).strip()) if str(value).strip() else -1
        )
        names = self._names_by_original_id(table)
        return {
            group_id: [
                {
                    "class_id": original_id,
                    "class_name": names[original_id],
                    "count": int((original_ids == original_id).sum()),
                }
                for original_id in members
            ]
            for group_id, members in self.class_groups.items()
        }

    def _filter_boxes(self, target_map: Dict[int, str]) -> None:
        """按几何条件筛选，再按类别映射或 ``class_groups`` 改写标签。"""
        table = self.sample_path_table
        masks = box_keep_masks(
            table,
            min_side_px=self.min_side_px,
            has_label=self._has_label,
            class_groups=self.class_groups,
            target_map=target_map,
        )
        keep = masks["keep"]
        filtered = table.loc[keep].reset_index(drop=True)
        member_counts = None
        if self._has_label and self.class_groups:
            merged_map = self._merged_class_map(table)
            original_to_group = {
                original_id: group_id
                for group_id, members in self.class_groups.items()
                for original_id in members
            }
            original_ids = filtered["class_id"].map(lambda value: int(str(value).strip()))
            filtered["class_id"] = original_ids.map(original_to_group).astype(int)
            filtered["class_name"] = filtered["class_id"].map(merged_map)
            empty_groups = sorted(set(merged_map) - set(filtered["class_id"].astype(int)))
            if empty_groups:
                raise ValueError(f"合并后这些组没有样本: {empty_groups}")
            target_map = merged_map
            member_counts = self._member_counts(table, keep)
        elif self._has_label and target_map:
            name_to_id = {name: class_id for class_id, name in target_map.items()}
            filtered["class_id"] = filtered["class_name"].map(name_to_id)
        self.sample_path_table = filtered
        self.num_samples = len(filtered)
        self.map_class_id_to_class_name = target_map
        if self._has_label and target_map and self.num_samples == 0:
            raise ValueError(
                "筛选后没有样本。检查 map_class_id_to_class_name 与 min_side_px"
            )
        self.kept_image_paths = set(filtered["img_path"].astype(str)) if len(filtered) else set()
        self.distribution = assemble_distribution(
            n_raw=len(table),
            masks=masks,
            kept=filtered,
            box_scale=self.box_scale,
            class_map=target_map if self._has_label else {},
            member_counts=member_counts,
            has_label=self._has_label,
        )

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
