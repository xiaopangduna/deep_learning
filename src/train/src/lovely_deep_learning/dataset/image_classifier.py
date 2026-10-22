# -*- encoding: utf-8 -*-

import csv
import random
import warnings
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
        绘图用的训练 id 到名称，不参与筛样本。有类别列时按筛选结果生成，构造参数里的映射被忽略。
        没有类别列时使用构造参数；没传则为空。
    class_groups : dict[int, list[int]] | None
        ``None`` 表示不按类别筛选。否则键是训练 id，值是要合并进来的原始 ``class_id``。
    background_id : int | None
        ``class_groups`` 里值为 ``background`` 的训练 id。整图分类不允许声明它。
    _has_label : bool
        表中有 ``class_id`` 列，且至少一行非空。

    Notes
    -----
    ``key_map`` 的每个值都必须是 CSV 表头，不会与表头求交。
    ``None`` 使用 :attr:`DEFAULT_KEY_MAP`，``{}`` 表示不重命名。
    预测表没有标签列时，传入只含图像路径的 ``key_map``。
    不传 ``class_groups`` 时，CSV 里出现的 ``class_id`` 全部保留，训练 id 和名字保持原值。
    分组里某个原始 id 在当前 CSV 中没有行时只警告。合并后的训练 id 在筛选后须至少有一行。
    """

    DEFAULT_KEY_MAP = {
        "img_path": "path_img",
        "class_name": "class_name",
        "class_id": "class_id",
    }
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)
    BACKGROUND_NAME = "background"

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        norm_mean: Optional[Sequence[float]] = _IMAGENET_MEAN,
        norm_std: Optional[Sequence[float]] = _IMAGENET_STD,
        class_groups: Optional[Dict[Any, Any]] = None,
        map_class_id_to_class_name: Optional[Dict[Any, str]] = None,
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
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，供 ``convert_img_from_tensor_to_numpy`` 反归一化。
            省略时使用 ImageNet 常数。``None`` 表示跳过均值方差反归一化。
        class_groups
            ``{训练 id: [原始 class_id, ...]}``。``None`` 或空字典表示全部训练。
            值为 ``background`` 的项是没有 CSV 行的背景类，整图分类会报错。
        map_class_id_to_class_name
            训练 id 到绘图名称。有类别列时忽略，名称按筛选结果生成。
            没有类别列时使用这份映射；``None`` 表示空映射。
        """
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)

        self.norm_mean = None if norm_mean is None else list(norm_mean)
        self.norm_std = None if norm_std is None else list(norm_std)
        self.class_groups, self.background_id = self.parse_class_groups(class_groups)
        self._given_class_names = self.normalize_class_name_map(map_class_id_to_class_name)
        self.map_class_id_to_class_name: Dict[int, str] = {}
        self._has_label = (
            "class_id" in self.sample_path_table.columns
            and len(self.sample_path_table) > 0
            and not self.sample_path_table["class_id"].astype(str).str.strip().eq("").all()
        )
        if not self._defers_class_filter():
            self._apply_class_groups()


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

    def class_name_for_id(self, class_id: int) -> str:
        """绘图用名称。映射里没有该 id 时返回 id 本身的字符串。"""
        return str(self.map_class_id_to_class_name.get(int(class_id), class_id))

    def _allows_background(self) -> bool:
        return False

    def _defers_class_filter(self) -> bool:
        return False

    def _apply_class_groups(self) -> None:
        """按 ``class_groups`` 改写标签，并生成绘图用的类别名。"""
        if self.background_id is not None and not self._allows_background():
            raise ValueError("整图分类没有框外区域，不能把 class_groups 的值写成 background")
        if not self._has_label:
            self.map_class_id_to_class_name = dict(self._given_class_names)
            return
        if self.class_groups is None:
            self.map_class_id_to_class_name = self.names_by_original_id(self.sample_path_table)
            return
        merged = self.group_names(self.sample_path_table, self.class_groups)
        filtered = self.rows_in_groups(self.sample_path_table, self.class_groups, merged)
        self.require_merged_samples(filtered, self.class_groups)
        self.sample_path_table = filtered
        self.map_class_id_to_class_name = merged

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
                # 反归一化: img * std + mean（通道数随 norm_mean/norm_std 长度，支持 1/3 通道）
                c = len(self.norm_mean)
                mean = torch.tensor(self.norm_mean).view(c, 1, 1)
                std = torch.tensor(self.norm_std).view(c, 1, 1)
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
    def normalize_class_name_map(class_names: Optional[Dict[Any, str]]) -> Dict[int, str]:
        """把构造参数收成 ``{int: str}``。``None`` 和空字典都是空映射。"""
        if not class_names:
            return {}
        if not isinstance(class_names, dict):
            raise TypeError(
                "map_class_id_to_class_name 须为 dict 或 None，"
                f"收到 {type(class_names)!r}"
            )
        return {int(class_id): str(name) for class_id, name in class_names.items()}

    @staticmethod
    def parse_class_groups(
        class_groups: Optional[Dict[Any, Any]],
    ) -> tuple[Optional[Dict[int, list[int]]], Optional[int]]:
        """返回 ``(分组, background 的训练 id)``。

        ``None`` 或空字典表示全部训练，背景 id 为 ``None``。
        值为 ``background`` 的键没有原始 id。其余值是原始 ``class_id`` 列表。
        分组键和背景 id 合在一起须为 ``0..max``。原始 id 不能重复，组不能为空。
        只有背景、没有原始 id 列表时，分组是空字典，表示不保留任何 CSV 类别。
        """
        if not class_groups:
            return None, None
        groups: Dict[int, list[int]] = {}
        background_id: Optional[int] = None
        background_name = ImageClassifierDataset.BACKGROUND_NAME
        for key, members in class_groups.items():
            group_id = int(key)
            if isinstance(members, str):
                if members != background_name:
                    raise ValueError(
                        f"class_groups[{group_id}] 的字符串只能是 {background_name}，收到 {members!r}"
                    )
                if background_id is not None:
                    raise ValueError(f"class_groups 里有多个 {background_name}")
                background_id = group_id
                continue
            if isinstance(members, (bytes, bytearray)) or not isinstance(members, Sequence):
                raise TypeError(
                    f"class_groups[{group_id}] 须为原始 class_id 列表或 {background_name}，收到 {members!r}"
                )
            original_ids = [int(member) for member in members]
            if not original_ids:
                raise ValueError(f"class_groups[{group_id}] 为空")
            if len(original_ids) != len(set(original_ids)):
                raise ValueError(f"class_groups[{group_id}] 含重复 id: {original_ids}")
            groups[group_id] = original_ids
        occupied = set(groups)
        if background_id is not None:
            occupied.add(background_id)
        last = max(occupied)
        expected = set(range(last + 1))
        if occupied != expected:
            detail = f"，背景 id {background_id}" if background_id is not None else ""
            raise ValueError(
                f"class_groups 的键须组成连续的 0..{last}，收到 {sorted(groups)}{detail}"
            )
        seen: Dict[int, int] = {}
        for group_id, original_ids in groups.items():
            for original_id in original_ids:
                if original_id in seen:
                    raise ValueError(
                        f"原始 class_id {original_id} 同时出现在组 {seen[original_id]} 和组 {group_id}"
                    )
                seen[original_id] = group_id
        return groups, background_id

    @staticmethod
    def names_by_original_id(table) -> Dict[int, str]:
        """原始 ``class_id`` 到 ``class_name``。同一个 id 出现两个名字则报错。"""
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

    @staticmethod
    def group_names(table, groups: Dict[int, list[int]]) -> Dict[int, str]:
        """分组训练 id 到用 ``+`` 拼起来的原名。

        某个原始 id 在当前表里没有行时只警告，名字里跳过它。
        合并后的训练 id 是否还有样本，由 :meth:`require_merged_samples` 在筛选之后检查。
        """
        names = ImageClassifierDataset.names_by_original_id(table)
        missing = sorted(
            {
                original_id
                for members in groups.values()
                for original_id in members
                if original_id not in names
            }
        )
        if missing:
            warnings.warn(
                f"class_groups 中的原始 class_id 在 CSV 里不存在: {missing}",
                UserWarning,
                stacklevel=2,
            )
        return {
            group_id: "+".join(
                names[original_id] for original_id in members if original_id in names
            )
            for group_id, members in groups.items()
        }

    @staticmethod
    def require_merged_samples(table, groups: Dict[int, list[int]]) -> None:
        """每个合并后的训练 id 在筛选后的表里至少有一行。"""
        if not groups:
            return
        present = set()
        if len(table) and "class_id" in table.columns:
            for value in table["class_id"]:
                text = str(value).strip()
                if text:
                    present.add(int(text))
        missing = sorted(group_id for group_id in groups if group_id not in present)
        if missing:
            raise ValueError(f"筛选后训练 id {missing} 没有样本。检查 class_groups")

    @staticmethod
    def rows_in_groups(table, groups: Dict[int, list[int]], merged: Dict[int, str]):
        """留下分组成员，并把 ``class_id``、``class_name`` 改成训练 id 和组名。"""
        if not groups:
            return table.iloc[0:0].reset_index(drop=True)
        original_to_group = {
            original_id: group_id
            for group_id, members in groups.items()
            for original_id in members
        }
        original_ids = table["class_id"].map(
            lambda value: int(str(value).strip()) if str(value).strip() else -1
        )
        keep = original_ids.isin(original_to_group)
        filtered = table.loc[keep].copy()
        filtered["class_id"] = original_ids.loc[keep].map(original_to_group).astype(int)
        filtered["class_name"] = filtered["class_id"].map(merged)
        return filtered.reset_index(drop=True)


class ImageClassifierFromDetectBoxDataset(ImageClassifierDataset):
    """一行一个框。CSV 里的 ``class_id`` 是原始 id。

    初始化时先按父类读入表，再筛选：丢掉 ``w`` 或 ``h`` 非正的框；短边
    ``min(w*img_w, h*img_h)`` 小于 ``min_side_px`` 的框。类别筛选与整图分类相同，
    由 ``class_groups`` 决定。值为 ``background`` 的训练 id 没有 CSV 行。

    ``background_ratio`` 大于 0 时，在正样本之后追加 ``round(正样本数 * background_ratio)``
    个虚拟样本：从有正样本的图像里均匀抽一张，裁块宽高从该图正样本框里抽，再按 ``box_scale`` 外扩。
    外扩后与该图任意有效框（含被类别筛掉的框）的 IoU 达到 ``background_iou`` 就重抽。
    ``0`` 表示不追加。验证、测试、预测保持 0。比例大于 0 时 ``class_groups`` 里必须有 ``background``。

    ``key_map`` 为 ``None`` 时使用带标签的 :attr:`DEFAULT_KEY_MAP`。
    预测表没有类别列时传入 :attr:`PREDICT_KEY_MAP`。
    ``box_scale`` 默认 1.2，裁剪前按中心外扩。缩放交给外部 ``transform``。

    筛选结果写在 ``distribution``（计数、类别、几何），只含 CSV 里留下的正样本。
    ``kept_image_paths`` 是保留框所属图像，供 DataModule 算 train/val/test 的路径交集。
    这里不打印，由 DataModule 在 ``fit`` 时汇总。
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
    _BACKGROUND_ATTEMPTS = 32

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        norm_mean: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_MEAN,
        norm_std: Optional[Sequence[float]] = ImageClassifierDataset._IMAGENET_STD,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
        class_groups: Optional[Dict[Any, Any]] = None,
        background_ratio: float = 0.0,
        background_iou: float = 0.3,
        map_class_id_to_class_name: Optional[Dict[Any, str]] = None,
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
            与 :class:`ImageClassifierDataset` 相同。值为 ``background`` 的训练 id 可以抽背景块。
        background_ratio
            相对正样本数的背景块比例。``0`` 表示不抽。大于 0 时 ``class_groups`` 里须有 ``background``。
        background_iou
            背景块外扩后与禁区框的 IoU 达到该值则重抽。范围 ``[0, 1]``。
        map_class_id_to_class_name
            交给父类。有类别列时忽略；预测表没有类别列时用它绘图。
        """
        if background_ratio < 0:
            raise ValueError(f"background_ratio 不能为负，收到 {background_ratio}")
        if not 0.0 <= background_iou <= 1.0:
            raise ValueError(f"background_iou 须在 [0, 1]，收到 {background_iou}")
        self.background_ratio = float(background_ratio)
        self.background_iou = float(background_iou)
        self._n_background = 0
        self._background_class_id = None
        self._background_class_name = None
        if key_map is None:
            key_map = dict(self.DEFAULT_KEY_MAP)
        super().__init__(
            csv_paths=csv_paths,
            key_map=key_map,
            transform=transform,
            norm_mean=norm_mean,
            norm_std=norm_std,
            class_groups=class_groups,
            map_class_id_to_class_name=map_class_id_to_class_name,
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
        self._filter_boxes()

    def _allows_background(self) -> bool:
        return True

    def _defers_class_filter(self) -> bool:
        return True

    def __len__(self):
        return len(self.sample_path_table) + self._n_background

    def __getitem__(self, index):
        n_positive = len(self.sample_path_table)
        if index >= n_positive:
            return self._background_item()
        row = self.sample_path_table.iloc[index]
        img_path = str(row["img_path"])
        img_np, img_shape = self.read_img(img_path, None)
        crop = self.crop_cxcywh(
            img_np,
            float(row["cx"]),
            float(row["cy"]),
            float(row["w"]),
            float(row["h"]),
            self.box_scale,
        )
        class_name = row["class_name"] if self._has_label else None
        class_id = int(row["class_id"]) if self._has_label else None
        return self._pack_crop(img_path, img_shape, crop, class_name, class_id)

    def _pack_crop(self, img_path, img_shape, crop, class_name, class_id):
        img_tv = tv_tensors.Image(BaseDataset.convert_img_from_numpy_to_tensor_uint8(crop))
        if self.transform:
            img_tv = self.transform(img_tv)
        net_in = {
            "img_path": img_path,
            "img_shape": img_shape,
            "img_tv_transformed": img_tv,
        }
        net_out: Dict[str, Any] = {}
        if class_name is not None:
            net_out["class_name"] = class_name
            net_out["class_id"] = class_id
        return net_in, net_out

    def _background_item(self):
        img_path, cx, cy, width, height = self._sample_background_box()
        img_np, img_shape = self.read_img(img_path, None)
        crop = self.crop_cxcywh(img_np, cx, cy, width, height, self.box_scale)
        return self._pack_crop(
            img_path,
            img_shape,
            crop,
            self._background_class_name,
            self._background_class_id,
        )

    def _sample_background_box(self):
        """从有正样本的图上抽一块，外扩后与该图禁区的 IoU 低于 ``background_iou``。"""
        if not self._positive_image_paths:
            raise RuntimeError("没有可抽背景的图像")
        for _ in range(self._BACKGROUND_ATTEMPTS):
            img_path = random.choice(self._positive_image_paths)
            width, height = self._positive_wh[img_path][random.randrange(len(self._positive_wh[img_path]))]
            width = float(width)
            height = float(height)
            scaled_w = width * self.box_scale
            scaled_h = height * self.box_scale
            if scaled_w >= 1.0 or scaled_h >= 1.0:
                cx, cy = 0.5, 0.5
            else:
                cx = random.uniform(scaled_w / 2.0, 1.0 - scaled_w / 2.0)
                cy = random.uniform(scaled_h / 2.0, 1.0 - scaled_h / 2.0)
            zones = self._exclusion_boxes.get(img_path)
            if zones is None or self.max_iou_cxcywh(cx, cy, scaled_w, scaled_h, zones) < self.background_iou:
                return img_path, cx, cy, width, height
        raise RuntimeError(
            f"尝试 {self._BACKGROUND_ATTEMPTS} 次仍无法裁出与禁区 IoU 低于 {self.background_iou} 的背景块"
        )

    def _configure_background(self, raw) -> None:
        if not self._has_label:
            if self.background_ratio > 0:
                raise ValueError("没有类别列时不能抽背景，background_ratio 须为 0")
            self._n_background = 0
            return
        if self.background_ratio == 0:
            self._n_background = 0
            return
        if self.background_id is None:
            raise ValueError(
                f"background_ratio 为 {self.background_ratio}，但 class_groups 里没有 {self.BACKGROUND_NAME}"
            )
        self._background_class_id = self.background_id
        self._background_class_name = self.BACKGROUND_NAME
        self._n_background = int(round(len(self.sample_path_table) * self.background_ratio))
        if self._n_background > 0:
            self._build_background_index(raw)

    def _build_background_index(self, raw) -> None:
        width = raw["w"].astype(float)
        height = raw["h"].astype(float)
        valid = raw.loc[(width > 0) & (height > 0)]
        exclusion: Dict[str, np.ndarray] = {}
        for img_path, group in valid.groupby(valid["img_path"].astype(str), sort=False):
            exclusion[str(img_path)] = group[["cx", "cy", "w", "h"]].astype(float).to_numpy(dtype=np.float32)
        self._exclusion_boxes = exclusion
        paths: list[str] = []
        sizes: Dict[str, np.ndarray] = {}
        filtered = self.sample_path_table
        for img_path, group in filtered.groupby(filtered["img_path"].astype(str), sort=False):
            key = str(img_path)
            paths.append(key)
            sizes[key] = group[["w", "h"]].astype(float).to_numpy(dtype=np.float32)
        self._positive_image_paths = paths
        self._positive_wh = sizes

    def _member_counts(self, table, keep) -> Dict[int, list]:
        kept = table.loc[keep]
        original_ids = kept["class_id"].map(
            lambda value: int(str(value).strip()) if str(value).strip() else -1
        )
        names = self.names_by_original_id(table)
        return {
            group_id: [
                {
                    "class_id": original_id,
                    "class_name": names.get(original_id, ""),
                    "count": int((original_ids == original_id).sum()),
                }
                for original_id in members
            ]
            for group_id, members in self.class_groups.items()
        }

    def _filter_boxes(self) -> None:
        """按几何条件筛选，再按 ``class_groups`` 改写标签。"""
        table = self.sample_path_table
        groups = self.class_groups if self._has_label else None
        masks = box_keep_masks(
            table,
            min_side_px=self.min_side_px,
            has_label=self._has_label,
            class_groups=groups,
        )
        keep = masks["keep"]
        filtered = table.loc[keep].reset_index(drop=True)
        member_counts = None
        if self._has_label and self.class_groups is not None:
            class_map = self.group_names(table, self.class_groups)
            filtered = self.rows_in_groups(filtered, self.class_groups, class_map)
            self.require_merged_samples(filtered, self.class_groups)
            member_counts = self._member_counts(table, keep)
        elif self._has_label and len(filtered):
            class_map = self.names_by_original_id(filtered)
        else:
            class_map = dict(self._given_class_names)
        if self._has_label and len(filtered) == 0:
            raise ValueError("筛选后没有样本。检查 class_groups 与 min_side_px")
        self.sample_path_table = filtered
        self.map_class_id_to_class_name = dict(class_map)
        if self.background_id is not None:
            self.map_class_id_to_class_name[self.background_id] = self.BACKGROUND_NAME
        self._n_background = 0
        self._configure_background(table)
        self.kept_image_paths = set(filtered["img_path"].astype(str)) if len(filtered) else set()
        self.distribution = assemble_distribution(
            n_raw=len(table),
            masks=masks,
            kept=filtered,
            box_scale=self.box_scale,
            class_map=class_map if self._has_label else {},
            member_counts=member_counts,
            has_label=self._has_label,
        )

    @staticmethod
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

    @staticmethod
    def max_iou_cxcywh(cx: float, cy: float, w: float, h: float, boxes: np.ndarray) -> float:
        """返回归一化 ``cxcywh`` 框与 ``boxes`` ``(N, 4)`` 的最大 IoU。"""
        if boxes.size == 0:
            return 0.0
        ax1, ay1 = cx - w / 2.0, cy - h / 2.0
        ax2, ay2 = cx + w / 2.0, cy + h / 2.0
        bx1 = boxes[:, 0] - boxes[:, 2] / 2.0
        by1 = boxes[:, 1] - boxes[:, 3] / 2.0
        bx2 = boxes[:, 0] + boxes[:, 2] / 2.0
        by2 = boxes[:, 1] + boxes[:, 3] / 2.0
        ix1 = np.maximum(ax1, bx1)
        iy1 = np.maximum(ay1, by1)
        ix2 = np.minimum(ax2, bx2)
        iy2 = np.minimum(ay2, by2)
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
        return float(np.max(inter / (area_a + area_b - inter + 1e-7)))
