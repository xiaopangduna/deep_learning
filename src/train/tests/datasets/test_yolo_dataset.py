from sys import prefix
from turtle import st

# from sympy import frac, fraction
from matplotlib.pylab import f
from ultralytics import YOLO
import yaml
# from ultralytics.models import yolo
# from ultralytics.utils import (
#     ARGV,
#     ASSETS,
#     DEFAULT_CFG_DICT,
#     LOGGER,
#     RANK,
#     SETTINGS,
#     YAML,
#     callbacks,
#     checks,
# )

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file


from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset


from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_cls_dataset, check_det_dataset

path_image = "/home/ubuntu/Desktop/project/deep_learning/src/train/datasets/coco8"

# model = YOLO("pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
# model.train(data="coco8.yaml", epochs=10,batch=1,workers=0)  # 在COCO数据集上训练模型


img_path = "/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/images/train"
imgsz = 640
batch_size = 1
augment = False

rect = False
cache = None
single_cls = False
stride = 32
pad = 0.0
prefix = colorstr("train: ")
task = "detect"
classes = None
fraction = 1.0

cfg = DEFAULT_CFG
overrides = {
    "task": "detect",
    "data": "coco8.yaml",
    "imgsz": 640,
    "single_cls": False,
    "model": "pretrained_models/yolov8n.pt",
    "epochs": 10,
    "batch": 1,
    "workers": 0,
    "mode": "train",
}
args = get_cfg(cfg, overrides)
hyp = cfg
data = check_det_dataset("coco8.yaml")
dataset = YOLODataset(
    img_path=img_path,
    imgsz=imgsz,
    batch_size=batch_size,
    augment=augment,
    hyp=hyp,
    rect=rect,
    cache=cache,
    single_cls=single_cls,
    stride=stride,
    pad=pad,
    prefix=prefix,
    task=task,
    classes=classes,
    data=data,
    fraction=fraction,
)
dataset[0]
pass
