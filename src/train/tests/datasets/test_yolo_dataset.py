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


# dataset_path = "/home/xiaopangdun/project/yolo/datasets/coco8/images/train"
# mode = "train"
# rank = -1
# batch_size = 1
# args = {
#     "task": "detect",
#     "data": "coco8.yaml",
#     "imgsz": 640,
#     "single_cls": False,
#     "model": "pretrained_models/yolov8n.pt",
#     "epochs": 10,
#     "batch": 1,
#     "mode": "train",
# }

# # self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
# # self.model = self.trainer.model
# my_callbacks = callbacks.get_default_callbacks()
# trainer =yolo.detect.DetectionTrainer(overrides=args, _callbacks=my_callbacks)
# trainer.model = trainer.get_model(weights=model.model if self.ckpt else None, cfg=self.model.yaml)
# dataloader = trainer.get_dataloader(dataset_path=dataset_path, mode=mode, rank=rank, batch_size=batch_size)
# for batch in dataloader:
#     pass
# pass

# def build_yolo_dataset(
#     cfg: IterableSimpleNamespace,
#     img_path: str,
#     batch: int,
#     data: Dict[str, Any],
#     mode: str = "train",
#     rect: bool = False,
#     stride: int = 32,
#     multi_modal: bool = False,
# ):
#     """Build and return a YOLO dataset based on configuration parameters."""
#     dataset = YOLODataset(
#         img_path=img_path,
#         imgsz=cfg.imgsz,
#         batch_size=batch,
#         augment=mode == "train",  # augmentation
#         hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
#         rect=cfg.rect or rect,  # rectangular batches
#         cache=cfg.cache or None,
#         single_cls=cfg.single_cls or False,
#         stride=int(stride),
#         pad=0.0 if mode == "train" else 0.5,
#         prefix=colorstr(f"{mode}: "),
#         task=cfg.task,
#         classes=cfg.classes,
#         data=data,
#         fraction=cfg.fraction if mode == "train" else 1.0,
#     )

# args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
# trainer = DetectionTrainer(overrides=args)
# trainer.train()

# dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
# dataset.get_labels()

img_path = "/home/ubuntu/Desktop/project/deep_learning/src/train/datasets/coco8/images/train"
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
