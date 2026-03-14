import pytest
import csv
import os
import numpy as np
import shutil
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from lovely_deep_learning.dataset.image_classifier import ImageClassifierDataset
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.utils import save_image
import torch

PATH_CSV = ["/home/xiaopangdun/project/deep_learning/src/train/datasets/IMAGENETTE/train.csv"]
KEY_MAP = {"img_path": "path_image", "class": "class", "class_id": "class_id"}


def test_ImageClassifierDataset_init():
    path_csv = PATH_CSV
    key_map = KEY_MAP
    dataset = ImageClassifierDataset(path_csv, key_map=key_map)


def test_ImageClassifierDataset_getitem():
    path_csv = PATH_CSV
    key_map = KEY_MAP
    dataset = ImageClassifierDataset(path_csv, key_map=key_map)
    net_in, net_out = dataset[0]

    pass


def test_ImageClassifierDataset_getitem_with_transform():
    path_csv = PATH_CSV
    key_map = KEY_MAP
    transforms = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(path_csv, key_map=key_map, transform=transforms)
    net_in, net_out = dataset[0]

    save_image(net_in["img_tv_transformed"], "./tmp/test_ImageClassifierDataset_getitem_with_transform.jpg")
