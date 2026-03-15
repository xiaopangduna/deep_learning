# -*- encoding: utf-8 -*-
"""
@File    :   classify.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/12 21:51:43
@Author  :   xiaopangdun
@Email   :   18675381281@163.com
@Desc    :   This is a simple example
"""
from typing import List, Dict, Optional, Callable, Any, Tuple, Union
import json
import warnings
from copy import deepcopy
import lightning as L
import torch
from torchvision.transforms import transforms as T

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from torchvision import tv_tensors

from .base import BaseDataset


class ImageClassifierDataset(BaseDataset):
    """A basic for building pytorch model input and output tensor

    This class inherits the Dataset(from torch.utils.data) to ensure the way of load dataset ,
    visualize data and label and data enhancemment is same.

    Args:
        path_txt (str): The path of txt file ,whose contents the paths of data and label.
        paths_data (list[str]): A list of paths of data.
        paths_label (list[str]):A list of paths of label.
        transfroms (str): One of train,val,test and  none.Default none
        cfgs (dict): A dictionary holds parameters in data processing.
    """

    def __init__(
        self,
        csv_paths: List[str],
        key_map: Dict[str, str] = {"img_path": "path_img", "class_name": "class_name", "class_id": "class_id"},
        transform: Optional[Callable] = None,
        map_class_id_to_class_name={0: "class_A", 1: "class_B"},
        norm_mean=None,
        norm_std=None
    ):
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.has_label = True if "class_id" in self.sample_path_table else False
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __getitem__(self, index):
        net_in, net_out = {}, {}
        img_path = self.sample_path_table["img_path"][index]

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
        if self.has_label:
            class_id = self.sample_path_table["class_id"][index]
            net_out["class_name"] = self.sample_path_table["class_name"][index]
            net_out["class_id"] = int(class_id)

        return net_in, net_out


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
        
        # 如果设置了归一化参数，则进行反归一化
        if self.norm_mean is not None and self.norm_std is not None:
            # 反归一化: img * std + mean
            mean = torch.tensor(self.norm_mean).view(3, 1, 1)
            std = torch.tensor(self.norm_std).view(3, 1, 1)
            img = img * std + mean
        
        # 确保值在[0,1]范围内
        img = torch.clamp(img, 0, 1)
        
        # 转换维度从(C, H, W)到(H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        
        # 转换到uint8格式 (0~1 -> 0~255)
        img_np = (img_np * 255).astype(np.uint8)
        
        # 从RGB转换为BGR（OpenCV格式）
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_np

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
            img = self.draw_label_on_numpy(img, class_name, class_id, color=color, pos=(5, 15))
            # 预测值
            img = self.draw_label_on_numpy(img, class_name_pred, class_id_pred, class_id_conf, color=color, pos=(5, 35))
        elif class_id != None:
            cv2.rectangle(img, (0, 0), (img.shape[1], 20), bg_color, -1)
            # 真值
            img = self.draw_label_on_numpy(img, class_name, class_id, color=color, pos=(5, 15))
        elif class_id_pred != None:
            cv2.rectangle(img, (0, 0), (img.shape[1], 20), bg_color, -1)
            # 预测值
            img = self.draw_label_on_numpy(img, class_name_pred, class_id_pred, class_id_conf, color=color, pos=(5, 15))
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
