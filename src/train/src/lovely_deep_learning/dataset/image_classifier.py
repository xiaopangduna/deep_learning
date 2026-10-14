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


class ImageClassifierDataset(BaseDataset):
    """整图分类数据集。一行一张图，返回 ``(net_in, net_out)``。

    ``net_in`` 含 ``img_path``、``img_shape``、``img_tv_transformed``。
    表中存在非空 ``class_id`` 时，``net_out`` 含 ``class_id`` 与 ``class_name``；否则 ``net_out`` 为空字典。

    Attributes
    ----------
    norm_mean, norm_std : list[float]
        可视化反归一化用的均值和方差。默认是 ImageNet 的值。
    map_class_id_to_class_name : dict[int, str]
        类别 id 到名称。未传入映射时为空字典。
    _has_label : bool
        表中有 ``class_id`` 列，且至少一行非空。

    Notes
    -----
    ``key_map`` 的每个值都必须是 CSV 表头，不会与表头求交。
    预测表没有标签列时，传入只含图像路径的 ``key_map``。
    """

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str] = {
            "img_path": "path_img", "class_name": "class_name", "class_id": "class_id"},
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
        norm_mean: list[float] = [0.485, 0.456, 0.406],
        norm_std: list[float] = [0.229, 0.224, 0.225],
    ):
        """
        Parameters
        ----------
        csv_paths
            至少一张已存在的 CSV。约定同 ``BaseDataset``。
        key_map
            键是读入后的列名，值是 CSV 表头。默认把 ``path_img``、``class_name``、``class_id``
            映成 ``img_path``、``class_name``、``class_id``。
        transform
            作用于 ``tv_tensors.Image`` 的变换。``None`` 表示不变换。
        map_class_id_to_class_name
            ``None`` 表示空映射。``dict`` 的键会转成 ``int``。``str`` 是 CSV 路径，
            表头须含 ``class_id``、``class_name``。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，供 ``convert_img_from_tensor_to_numpy`` 反归一化。
        """
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)

        self.norm_mean = norm_mean
        self.norm_std = norm_std
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