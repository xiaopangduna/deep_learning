# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Callable

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
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
    ):
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self._has_label = "object_label_path" in self.sample_path_table.columns

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        net_in, net_out = {}, {}
        img_path = str(self.sample_path_table["img_path"].iloc[index])
        img_np, img_shape = BaseDataset.read_img(img_path, None)
        img_tensor = BaseDataset.convert_img_from_numpy_to_tensor_uint8(img_np)
        img_tv = tv_tensors.Image(img_tensor)
        if self.transform:
            img_tv_transformed = self.transform(img_tv)
        else:
            img_tv_transformed = img_tv
        if self._has_label:
            object_label_path = str(self.sample_path_table["object_label_path"].iloc[index])
            pass

        net_in["img_path"] = img_path
        net_in["img_shape"] = img_shape
        net_in["img_tv_transformed"] = img_tv_transformed

        return net_in, net_out
