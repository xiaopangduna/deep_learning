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
        key_map: Dict[str, str] = {"img_path": "path_image", "class": "class", "class_id": "class_id"},
        transform: Optional[Callable] = None,
    ):
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.has_label = self.has_label = "class_id" in self.sample_path_table

    def __getitem__(self, index):
        net_in, net_out = {}, {}
        img_path = self.sample_path_table["img_path"][index]

        img_np, img_shape = self.read_img(img_path, None)
        img_tensor = self.convert_img_from_numpy_to_tensor(img_np)
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
            net_out["class_name"] = self.sample_path_table["class"][index]
            net_out["class_id"] = int(class_id)

        return net_in, net_out

    # def draw_tensor_on_image_with_label(self, image: np.ndarray, output, target=None):
    #     # 绘制预测结果和真值在同一图像上
    #     class_to_index = self.cfgs["class_to_index"]
    #     _, index = torch.max(output, dim=0)
    #     _, target_index = torch.max(target, dim=0)
    #     for key in class_to_index.keys():
    #         if class_to_index[key] == index.item():
    #             class_perdict = key
    #         if class_to_index[key] == target_index.item():
    #             class_target = key
    #     if class_perdict != class_target:
    #         color = (0, 0, 255)
    #     else:
    #         color = (0, 255, 0)
    #     image = np.ascontiguousarray(image)
    #     cv2.putText(image, "pred: " + class_perdict, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    #     cv2.putText(image, "true: " + class_target, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    #     return image

    # @staticmethod
    # def draw_predict_on_image_without_label(image: np.ndarray, output, cfgs: dict):
    #     # 绘制预测结果和真值在同一图像上
    #     class_to_index = cfgs["class_to_index"]
    #     _, index = torch.max(output, dim=0)
    #     color = (0, 255, 0)
    #     for key in class_to_index.keys():
    #         if class_to_index[key] == index.item():
    #             class_perdict = key
    #     image = np.ascontiguousarray(image)
    #     cv2.putText(image, "pred: " + class_perdict, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    #     return image

    # def get_visual_grid_for_train(self, model):
    #     visual_indexs = random.sample(range(0, len(self)), self.cfgs["visual_nums"])
    #     row = col = int(pow(self.cfgs["visual_nums"], 1 / 2))
    #     images = []
    #     for i in visual_indexs:
    #         net_in, net_out = self[i]
    #         data = net_in["data"]
    #         data_tensor = net_in["data_tensor"].cuda()
    #         targer_tesnsor = net_out["label_tensor"]
    #         model.cuda()
    #         output = model(data_tensor.unsqueeze(0))
    #         image = self.draw_tensor_on_image_with_label(data, output.squeeze(), targer_tesnsor)
    #         image = cv2.resize(image, (300, 300))
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         images.append(image)
    #     fig, axs = plt.subplots(row, col)
    #     for i in range(row):
    #         for j in range(col):
    #             axs[i, j].imshow(images[i + j])
    #             axs[i, j].axis("off")
    #     # plt.tight_layout()
    #     # plt.savefig("bb.jpg")
    #     # for i in range(9):
    #     #     image = images[i]
    #     #     cv2.imwrite(r"{}.jpg".format(i),image)
    #     return fig

    # @staticmethod
    # def get_collate_fn_for_dataloader():
    #     def collate_fn(x):
    #         return list(zip(*x))

    #     return collate_fn
