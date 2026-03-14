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
PATH_CSV_WITHOUT_LABEL = ["datasets/IMAGENETTE/predict.csv"]
map_class_id_to_class_name = {
    0: "n01440764",
    1: "n02102040",
    2: "n02979186",
    3: "n03000684",
    4: "n03028079",
    5: "n03394916",
    6: "n03417042",
    7: "n03425413",
    8: "n03445777",
    9: "n03888257",
}


def test_ImageClassifierDataset_init():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(path_csv)


def test_ImageClassifierDataset__init_without_label():
    path_csv = PATH_CSV_WITHOUT_LABEL
    dataset = ImageClassifierDataset(path_csv)
    pass


def test_ImageClassifierDataset_getitem():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(path_csv)
    net_in, net_out = dataset[0]

    pass


def test_ImageClassifierDataset_getitem_with_transform():
    path_csv = PATH_CSV
    transforms = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(path_csv, transform=transforms)
    net_in, net_out = dataset[0]

    save_image(net_in["img_tv_transformed"], "./tmp/test_ImageClassifierDataset_getitem_with_transform.jpg")


def test_ImageClassifierDataset_getitem_with_transform_without_label():
    path_csv = PATH_CSV_WITHOUT_LABEL
    transforms = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(path_csv, transform=transforms)
    net_in, net_out = dataset[0]

    save_image(
        net_in["img_tv_transformed"], "./tmp/test_ImageClassifierDataset_getitem_with_transform_without_label.jpg"
    )


def test_ImageClassifierDataset_draw_label_on_numpy():
    path_csv = PATH_CSV
    transforms = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(path_csv, transform=transforms)
    net_in, net_out = dataset[0]
    img = cv2.imread(net_in["img_path"])
    img = cv2.resize(img, (224, 224))
    img_with_label = dataset.draw_label_on_numpy(img, net_out["class_name"], net_out["class_id"])
    cv2.imwrite("./tmp/test_ImageClassifierDataset_draw_label_on_numpy.jpg", img_with_label)
    pass


def test_ImageClassifierDataset_draw_target_and_predict_label_on_numpy():
    path_csv = PATH_CSV
    transforms = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(
        path_csv, transform=transforms, map_class_id_to_class_name=map_class_id_to_class_name
    )
    net_in, net_out = dataset[0]
    img = cv2.imread(net_in["img_path"])
    img = cv2.resize(img, (224, 224))
    img_with_label = dataset.draw_target_and_predict_label_on_numpy(
        img, net_out["class_name"], net_out["class_id"], class_name_pred="test", class_id_pred=0, class_id_conf=95.5
    )
    cv2.imwrite("./tmp/test_ImageClassifierDataset_draw_target_and_predict_label_on_numpy.jpg", img_with_label)
    pass
