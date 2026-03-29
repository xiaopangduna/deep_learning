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

PATH_CSV = ["tests/test_data/dataset/test_image_classifier_train.csv"]
PATH_CSV_WITHOUT_LABEL = [
    "tests/test_data/dataset/test_image_classifier_predict.csv"]

KEY_MAP = {"img_path": "path_img",
           "class_name": "class_name", "class_id": "class_id"}

MAP_CLASS_ID_TO_CLASS_NAME = {
    0: "n01440764",
    1: "n02102040",
}

MAP_CSV_PATH = "tests/test_data/dataset/map_class_id_to_class_name.csv"


def test_ImageClassifierDataset_init():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(
        path_csv, KEY_MAP, None, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    assert len(dataset.sample_path_table) == 2
    assert "img_path" in dataset.sample_path_table.columns
    assert "class_name" in dataset.sample_path_table.columns
    assert "class_id" in dataset.sample_path_table.columns


def test_ImageClassifierDataset_init_with_map_csv_path():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(
        path_csv, KEY_MAP, None, MAP_CSV_PATH, None, None)
    assert dataset.map_class_id_to_class_name == MAP_CLASS_ID_TO_CLASS_NAME


def test_ImageClassifierDataset_init_without_label():
    path_csv = PATH_CSV_WITHOUT_LABEL
    dataset = ImageClassifierDataset(
        path_csv, {"img_path": "path_img"}, None, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    assert "img_path" in dataset.sample_path_table.columns
    assert "class_name" not in dataset.sample_path_table.columns
    assert "class_id" not in dataset.sample_path_table.columns

# /home/xiaopangdun/project/deep_learning/src/train/tests/test_data/dataset/base_img/ILSVRC2012_val_00000293.JPEG


def test_ImageClassifierDataset_getitem_with_transform():
    path_csv = PATH_CSV
    transforms = v2.Compose(
        [v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(
        path_csv, KEY_MAP, transforms, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    net_in, net_out = dataset[0]
    assert net_in["img_tv_transformed"].shape == (3, 224, 224)
    assert net_out["class_id"] == 0

    # save_image(net_in["img_tv_transformed"],
    #            "./tmp/test_ImageClassifierDataset_getitem_with_transform.jpg")


def test_ImageClassifierDataset_getitem_with_transform_without_label():
    path_csv = PATH_CSV_WITHOUT_LABEL
    transforms = v2.Compose(
        [v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ImageClassifierDataset(
        path_csv,  {"img_path": "path_img"}, transforms, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    net_in, net_out = dataset[0]

    assert net_in["img_tv_transformed"].shape == (3, 224, 224)

    # save_image(
    #     net_in["img_tv_transformed"], "./tmp/test_ImageClassifierDataset_getitem_with_transform_without_label.jpg"
    # )


def test_ImageClassifierDataset_draw_label_on_numpy():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(
        path_csv, KEY_MAP, None, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    net_in, net_out = dataset[0]
    img = cv2.imread(net_in["img_path"])
    img = cv2.resize(img, (224, 224))
    img_with_label = dataset.draw_label_on_numpy(
        img, net_out["class_name"], net_out["class_id"])

    # cv2.imwrite(
    #     "./tmp/test_ImageClassifierDataset_draw_label_on_numpy.jpg", img_with_label)


def test_ImageClassifierDataset_draw_target_and_predict_label_on_numpy():
    path_csv = PATH_CSV
    dataset = ImageClassifierDataset(
        path_csv, KEY_MAP, None, MAP_CLASS_ID_TO_CLASS_NAME, None, None)
    net_in, net_out = dataset[0]
    img = cv2.imread(net_in["img_path"])
    img = cv2.resize(img, (224, 224))
    img_with_label = dataset.draw_target_and_predict_label_on_numpy(
        img, net_out["class_name"], net_out["class_id"], class_name_pred="test", class_id_pred=0, class_id_conf=95.5
    )
    # cv2.imwrite(
    #     "./tmp/test_ImageClassifierDataset_draw_target_and_predict_label_on_numpy.jpg", img_with_label)


def test_load_map_class_id_to_class_name_from_csv():
    m = ImageClassifierDataset.load_map_class_id_to_class_name_from_csv(
        MAP_CSV_PATH)
    assert m == MAP_CLASS_ID_TO_CLASS_NAME
