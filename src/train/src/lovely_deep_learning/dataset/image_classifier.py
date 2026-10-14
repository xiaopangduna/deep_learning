# -*- encoding: utf-8 -*-

import csv
from typing import List, Dict, Optional, Callable, Any, Union, Sequence
from pathlib import Path

import torch
from torchvision.transforms import transforms as T
import numpy as np
import cv2
from torchvision import tv_tensors

from .base import BaseDataset
from .box_crop_distribution import assemble_distribution, box_keep_masks


class ImageClassifierDataset(BaseDataset):
    """整图分类数据集。一行一张图，返回 ``(net_in, net_out)``。

    ``net_in`` 含 ``img_path``、``img_shape``、``img_tv_transformed``。
    表中存在非空 ``class_id`` 时，``net_out`` 含 ``class_id`` 与 ``class_name``；否则 ``net_out`` 为空字典。

    Attributes
    ----------
    norm_mean, norm_std : list[float] | None
        可视化反归一化用的均值和方差。省略时各实例得到一份 ImageNet 常数的新列表。
        ``None`` 表示不做均值方差反归一化。
    map_class_id_to_class_name : dict[int, str]
        类别 id 到名称。未传入映射时为空字典。
    _has_label : bool
        表中有 ``class_id`` 列，且至少一行非空。

    Notes
    -----
    ``key_map`` 的每个值都必须是 CSV 表头，不会与表头求交。
    ``None`` 使用 :attr:`DEFAULT_KEY_MAP`，``{}`` 表示不重命名。
    预测表没有标签列时，传入只含图像路径的 ``key_map``。
    """

    DEFAULT_KEY_MAP = {
        "img_path": "path_img",
        "class_name": "class_name",
        "class_id": "class_id",
    }
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
        norm_mean: Optional[Sequence[float]] = _IMAGENET_MEAN,
        norm_std: Optional[Sequence[float]] = _IMAGENET_STD,
    ):
        """
        Parameters
        ----------
        csv_paths
            至少一张已存在的 CSV。约定同 ``BaseDataset``。
        key_map
            键是读入后的列名，值是 CSV 表头。``None`` 复制 :attr:`DEFAULT_KEY_MAP`：
            ``path_img`` 映成 ``img_path``，``class_name`` 与 ``class_id`` 列名不变。
            ``{}`` 表示不重命名。
        transform
            作用于 ``tv_tensors.Image`` 的变换。``None`` 表示不变换。
        map_class_id_to_class_name
            ``None`` 表示空映射。``dict`` 的键会转成 ``int``。``str`` 是 CSV 路径，
            表头须含 ``class_id``、``class_name``。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，供 ``convert_img_from_tensor_to_numpy`` 反归一化。
            省略时使用 ImageNet 常数。``None`` 表示跳过均值方差反归一化。
        """
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)

        self.norm_mean = None if norm_mean is None else list(norm_mean)
        self.norm_std = None if norm_std is None else list(norm_std)
        self.map_class_id_to_class_name = self._normalize_map_class_id_to_class_name(
            map_class_id_to_class_name
        )

        self._has_label = (
            "class_id" in self.sample_path_table.columns
            and len(self.sample_path_table) > 0
            and not self.sample_path_table["class_id"].astype(str).str.strip().eq("").all()
        )


    def __getitem__(self, index):
        net_in, net_out = {}, {}
        img_path = str(self.sample_path_table["img_path"].iloc[index])

        img_np, img_shape = self.read_img(img_path, None)
        img_tensor = BaseDataset.convert_img_from_numpy_to_tensor_uint8(img_np)
        img_tv = tv_tensors.Image(img_tensor)
        if self.transform:
            img_tv_transformed = self.transform(img_tv)
        else:
            img_tv_transformed = img_tv
        net_in["img_path"] = img_path
        net_in["img_shape"] = img_shape
        net_in["img_tv_transformed"] = img_tv_transformed
        if self._has_label:
            class_id = self.sample_path_table["class_id"].iloc[index]
            net_out["class_name"] = self.sample_path_table["class_name"].iloc[index]
            net_out["class_id"] = int(class_id)

        return net_in, net_out
    
    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(samples):
            net_in = BaseDataset.collate_net_in_tuple(samples)
            net_out_tuple = tuple(no for _, no in samples)
            if not net_out_tuple or all(not no for no in net_out_tuple):
                return net_in, {}
            if "class_id" not in net_out_tuple[0]:
                return net_in, {}
            return net_in, {
                "class_id": torch.tensor(
                    [no["class_id"] for no in net_out_tuple], dtype=torch.long
                ),
                "class_name": [no["class_name"] for no in net_out_tuple],
            }

        return collate_fn

    def convert_img_from_tensor_to_numpy(self, img: torch.Tensor) -> np.ndarray:
        """
        将标准化的tensor转换为uint8的numpy数组，并进行反标准化处理

        Args:
            img: 输入的标准化tensor，格式为(C, H, W)

        Returns:
            反标准化后的numpy数组，格式为(H, W, C)，值域为[0, 255]
        """
        # 获取设备信息并复制tensor到CPU
        img = img.detach().cpu()

        # 获取tensor的值范围
        img_min = img.min().item()
        img_max = img.max().item()

        # 根据值的范围判断tensor类型并进行相应处理
        if img_min >= 0 and img_max <= 1.0:
            # 情况1: [0, 1] 范围的tensor
            img_clamped = torch.clamp(img, 0, 1)
        elif img_min >= 0 and img_max <= 255:
            # 情况2: [0, 255] 范围的tensor
            img_clamped = torch.clamp(img, 0, 255) / 255.0  # 归一化到[0,1]
        else:
            # 情况3: 标准化的tensor，需要反标准化
            if self.norm_mean is not None and self.norm_std is not None:
                # 反归一化: img * std + mean
                mean = torch.tensor(self.norm_mean).view(3, 1, 1)
                std = torch.tensor(self.norm_std).view(3, 1, 1)
                img = img * std + mean
            # 确保值在合理范围内
            img_clamped = torch.clamp(img, 0, 1)

        # 将[0,1]范围的值转换为uint8格式 (0~1 -> 0~255)
        img_uint8 = (img_clamped * 255).to(torch.uint8)

        # 调用基类的转换函数
        return BaseDataset.convert_img_from_tensor_to_numpy_uint8(img_uint8)

    def draw_target_and_predict_label_on_numpy(
        self,
        img: np.ndarray,
        class_name: str = None,
        class_id: int = None,
        class_name_pred: str = None,
        class_id_pred: int = None,
        class_id_conf: float = None,
    ):
        color = (0, 255, 0)
        bg_color = (255, 255, 255)
        if class_id != None and class_id_pred != None:
            # 预测值和真值一致
            if class_id != class_id_pred:
                color = (0, 0, 255)
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), bg_color, -1)
            # 真值
            img = self.draw_label_on_numpy(
                img, class_name, class_id, color=color, pos=(5, 15))
            # 预测值
            img = self.draw_label_on_numpy(
                img, class_name_pred, class_id_pred, class_id_conf, color=color, pos=(5, 35))
        elif class_id != None:
            cv2.rectangle(img, (0, 0), (img.shape[1], 20), bg_color, -1)
            # 真值
            img = self.draw_label_on_numpy(
                img, class_name, class_id, color=color, pos=(5, 15))
        elif class_id_pred != None:
            cv2.rectangle(img, (0, 0), (img.shape[1], 20), bg_color, -1)
            # 预测值
            img = self.draw_label_on_numpy(
                img, class_name_pred, class_id_pred, class_id_conf, color=color, pos=(5, 15))
        else:
            pass
        return img

    def draw_label_on_numpy(
        self,
        img: np.ndarray,
        class_name: str = "",
        class_id: int = None,
        class_id_conf: float = None,
        color=(0, 255, 0),
        pos=(5, 15),
        font_scale=0.5,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        if class_id_conf != None:
            text = f"pred   id:{class_id:03d} c:{class_id_conf:.1f} n:{class_name:<15} "
        else:
            text = f"target id:{class_id:03d} n:{class_name:<15}"
        cv2.putText(img, text, pos, font, font_scale, color, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def load_map_class_id_to_class_name_from_csv(path: Union[str, Path]) -> Dict[int, str]:
        """读取表头为 class_id, class_name 的 CSV，返回 ``{int: str}``。"""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"类别映射 CSV 不存在：{p}")
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV 无表头：{p}")
            lower = {name.lower(): name for name in reader.fieldnames}
            if "class_id" not in lower or "class_name" not in lower:
                raise ValueError(
                    f"CSV 须包含 class_id、class_name 列（表头不区分大小写）：{p}"
                )
            id_col, name_col = lower["class_id"], lower["class_name"]
            out: Dict[int, str] = {}
            for row in reader:
                raw_id = row.get(id_col)
                if raw_id is None or str(raw_id).strip() == "":
                    continue
                cid = int(str(raw_id).strip())
                name = str(row.get(name_col, "")).strip()
                if cid in out and out[cid] != name:
                    raise ValueError(
                        f"同一 class_id 在映射 CSV 中对应多个 class_name：id={cid} "
                        f"{out[cid]!r} vs {name!r}（{p}）"
                    )
                out[cid] = name
            return out

    @staticmethod
    def _normalize_map_class_id_to_class_name(
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]],
    ) -> Dict[int, str]:
        if map_class_id_to_class_name is None:
            return {}
        if isinstance(map_class_id_to_class_name, str):
            return ImageClassifierDataset.load_map_class_id_to_class_name_from_csv(
                map_class_id_to_class_name
            )
        if isinstance(map_class_id_to_class_name, dict):
            return {
                int(k): str(v)
                for k, v in map_class_id_to_class_name.items()
            }
        raise TypeError(
            "map_class_id_to_class_name 须为 None、dict 或 str（CSV 路径），"
            f"收到 {type(map_class_id_to_class_name)!r}"
        )

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


class ImageClassifierFromDetectBoxDataset(ImageClassifierDataset):
    """一行一个框。CSV 里的 ``class_id`` 是原始 id。

    初始化时先按父类读入表，再筛选：丢掉 ``w`` 或 ``h`` 非正的框；短边
    ``min(w*img_w, h*img_h)`` 小于 ``min_side_px`` 的框。有标签时：

    - 传入 ``class_groups`` 则按组合并。键是合并后的训练 id（须为 ``0..n-1``），
      值是原始 ``class_id``。未出现在任何组里的原始类丢掉。样本的 ``class_id`` 改成组号，
      ``class_name`` 改成组内原始类名用 ``+`` 拼接，并写入 ``map_class_id_to_class_name``。
    - 未传 ``class_groups`` 且类别映射非空时，只保留映射里的类别名，并把 ``class_id`` 改写成该映射的 id。
    - 未传 ``class_groups`` 且类别映射为空时，只做几何筛选，``class_id`` 保持 CSV 中的原始值。

    ``key_map`` 为 ``None`` 时使用带标签的 :attr:`DEFAULT_KEY_MAP`。
    预测表没有类别列时传入 :attr:`PREDICT_KEY_MAP`。
    ``box_scale`` 默认 1.2，裁剪前按中心外扩。缩放交给外部 ``transform``。

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
        norm_mean: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_MEAN,
        norm_std: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_STD,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
        class_groups: Optional[Dict[Any, Sequence[int]]] = None,
    ):
        """
        Parameters
        ----------
        key_map
            ``None`` 复制 :attr:`DEFAULT_KEY_MAP`。预测表没有类别列时传入 :attr:`PREDICT_KEY_MAP`。
        norm_mean, norm_std
            交给父类。省略时使用 ImageNet 常数。``None`` 表示跳过均值方差反归一化。
        box_scale
            裁剪前按中心外扩的倍数，须为正数。
        min_side_px
            短边小于该像素的框丢掉。``0`` 表示不按短边筛选。
        class_groups
            ``{训练 id: [原始 class_id, ...]}``。``None`` 或空字典表示不合并。
        """
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
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
        self.class_groups = normalize_class_groups(class_groups)
        self._filter_boxes(target_map)

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
        self.map_class_id_to_class_name = target_map
        if self._has_label and target_map and len(self) == 0:
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
