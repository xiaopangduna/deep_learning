# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Callable, List

import numpy as np
import pandas as pd
import torch
import cv2
from torchvision import tv_tensors

from .base import BaseDataset


class ObjectDetectDataset(BaseDataset):

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str] = {"img_path": "path_img", "object_label_path": "path_label_detect_yolo"},
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
    ):
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self.norm_mean = None
        self.norm_std = None
        self._has_label = "object_label_path" in self.sample_path_table.columns

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        net_in, net_out = {}, {}
        img_path = str(self.sample_path_table["img_path"].iloc[index])
        img_np, img_shape = BaseDataset.read_img(img_path, None)
        img_tensor = BaseDataset.convert_img_from_numpy_to_tensor_uint8(img_np)
        img_tv = tv_tensors.Image(img_tensor)
        if not self._has_label:
            if self.transform:
                img_tv_transformed = self.transform(img_tv)
            else:
                img_tv_transformed = img_tv
            net_in["img_path"] = img_path
            net_in["img_shape"] = img_shape
            net_in["img_tv_transformed"] = img_tv_transformed
            return net_in, net_out
        else:
            object_label_path = str(
                self.sample_path_table["object_label_path"].iloc[index])
            cls, bboxes_cxcywh_rel = ObjectDetectDataset.read_yolo_detection_labels(
                object_label_path
            )
            bboxes_abs_xyxy = ObjectDetectDataset.convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
                bboxes_cxcywh_rel, img_shape
            )
            cls_tensor = torch.from_numpy(cls)
            bboxes_tensor = torch.from_numpy(
                bboxes_abs_xyxy.astype(np.float32, copy=False)
            )
            bboxes_tv = tv_tensors.BoundingBoxes(
                bboxes_tensor,
                format="XYXY",
                canvas_size=(img_shape[0], img_shape[1]),
            )
            if self.transform:
                target = {
                    "cls": cls_tensor,
                    "bboxes": bboxes_tv
                }
                img_tv_transformed, target_transformed = self.transform(
                    img_tv, target)
                cls_tensor_transformed = target_transformed["cls"]
                bboxes_tv_transformed = target_transformed["bboxes"]
            else:
                img_tv_transformed = img_tv
                cls_tensor_transformed = cls_tensor
                bboxes_tv_transformed = bboxes_tv
            net_in["img_path"] = img_path
            net_in["img_shape"] = img_shape
            net_in["img_tv_transformed"] = img_tv_transformed
            net_out["cls_np"] = cls
            net_out["bboxes_cxcywh_rel_np"] = bboxes_cxcywh_rel
            net_out["cls_tv_transformed"] = cls_tensor_transformed
            net_out["bboxes_xyxy_abs_tv_transformed"] = bboxes_tv_transformed
        return net_in, net_out

    @staticmethod
    def read_yolo_detection_labels(file_path: str):
        """
        从YOLO格式的.txt文件中读取目标检测标签，并转换为NumPy数组。

        此函数假设标签文件格式正确，不包含任何校验。

        Args:
            file_path (str): YOLO标签文件的路径。

        Returns:
            cls: ``(N,)`` int32，类别 id。
            bbox: ``(N, 4)`` float32，YOLO txt 原始归一化 **CXCYWH**（相对宽高）。
        """
        if not file_path:
            labels = np.empty((0, 5), dtype=np.float32)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                labels = [x.split()
                          for x in f.read().strip().splitlines() if len(x)]
                labels = np.array(labels, dtype=np.float32)

        cls = labels[:, 0].astype(np.int32)
        bbox = labels[:, 1:5].astype(np.float32)
        return cls, bbox

    @staticmethod
    def convert_cxcywh_relative_to_absolute(
        bboxes_rel: np.ndarray, img_shape: tuple
    ) -> np.ndarray:
        """
        归一化 CXCYWH → 像素 CXCYWH（相对宽高分别乘 ``W``、``H``）。
        """
        H, W = img_shape[0], img_shape[1]
        scale = np.array([W, H, W, H], dtype=np.float32)
        return np.asarray(bboxes_rel, dtype=np.float32) * scale

    @staticmethod
    def convert_cxcywh_to_xyxy_absolute(bboxes_abs_cxcywh: np.ndarray) -> np.ndarray:
        """
        像素 CXCYWH → 像素 XYXY。
        """
        t = np.asarray(bboxes_abs_cxcywh, dtype=np.float32)
        x_c, y_c, bw, bh = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        x1 = x_c - bw / 2
        y1 = y_c - bh / 2
        x2 = x_c + bw / 2
        y2 = y_c + bh / 2
        return np.column_stack((x1, y1, x2, y2))

    @staticmethod
    def convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
        bboxes_rel: np.ndarray, img_shape: tuple
    ) -> np.ndarray:
        """
        归一化 CXCYWH → 像素 XYXY（等价于先
        ``convert_cxcywh_relative_to_absolute`` 再 ``convert_cxcywh_absolute_to_xyxy``）。
        """
        abs_cxcywh = ObjectDetectDataset.convert_cxcywh_relative_to_absolute(
            bboxes_rel, img_shape
        )
        return ObjectDetectDataset.convert_cxcywh_to_xyxy_absolute(abs_cxcywh)



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

    @staticmethod
    def draw_label_on_numpy(
        image: np.ndarray,
        bboxes: np.ndarray,
        classes: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = [(0, 255, 0)],
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制边界框（**仅支持像素 XYXY**）。

        参数:
            image: BGR 图像，``(H, W, C)``
            bboxes: ``(N, 4)``，每行 ``x1, y1, x2, y2`` 像素坐标，与 ``image`` 尺寸一致
            classes: 类别 ID，长度 ``N``
            class_names: 可选 id→名称
            colors: BGR 颜色列表
            thickness: 线宽

        返回:
            绘制后的图像（就地修改 ``image`` 并返回同一数组）
        """
        h_img, w_img = image.shape[:2]

        if colors is None:
            colors = [(0, 255, 0)]

        for i, (bbox, cls_id) in enumerate(zip(bboxes, classes)):
            x1, y1, x2, y2 = (float(bbox[j]) for j in range(4))
            x1 = int(round(max(0, min(x1, w_img - 1))))
            x2 = int(round(max(0, min(x2, w_img - 1))))
            y1 = int(round(max(0, min(y1, h_img - 1))))
            y2 = int(round(max(0, min(y2, h_img - 1))))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # 获取当前框的颜色
            if len(colors) > 1:
                color = colors[cls_id % len(colors)]
            else:
                color = colors[0]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # 添加类别标签
            label = f"Class: {cls_id}"
            if class_names and cls_id in class_names:
                label = class_names[cls_id]

            # 计算标签背景框的大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

            # 绘制标签背景
            cv2.rectangle(
                image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

            # 绘制标签文字
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),  # 白色文字
                1,
                cv2.LINE_AA
            )

        return image
