import pytest
import csv
import os
import numpy as np
import shutil
import cv2
from pathlib import Path

from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.utils import save_image
import torch

from lovely_deep_learning.data_module.image_classifier import ImageClassifierDataModule

PATH_TRAIN_CSV_PATHS = ["./datasets/IMAGENETTE/train.csv"]
PATH_PREDICT_CSV_PATHS = ["./datasets/IMAGENETTE/predict.csv"]

KEY_MAP = {"img_path": "path_img", "class_name": "class_name", "class_id": "class_id"}
BATCH_SIZE = 1
NUM_WORKERS = 1
TRANSFORM_TRAIN = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])

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


def test_ImageClassifierDataModule_init():

    data_module = ImageClassifierDataModule(
        train_csv_paths=PATH_TRAIN_CSV_PATHS,
        val_csv_paths=PATH_TRAIN_CSV_PATHS,
        test_csv_paths=PATH_TRAIN_CSV_PATHS,
        predict_csv_paths=PATH_PREDICT_CSV_PATHS,
        key_map=KEY_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform_train=TRANSFORM_TRAIN,
        transform_val=TRANSFORM_TRAIN,
        transform_test=TRANSFORM_TRAIN,
        transform_predict=TRANSFORM_TRAIN,
        map_class_id_to_class_name=map_class_id_to_class_name,
    )

    pass


def test_ImageClassifierDataModule_setup():
    data_module = ImageClassifierDataModule(
        train_csv_paths=PATH_TRAIN_CSV_PATHS,
        val_csv_paths=PATH_TRAIN_CSV_PATHS,
        test_csv_paths=PATH_TRAIN_CSV_PATHS,
        predict_csv_paths=PATH_PREDICT_CSV_PATHS,
        transform_train=TRANSFORM_TRAIN,
        transform_val=TRANSFORM_TRAIN,
        transform_test=TRANSFORM_TRAIN,
        transform_predict=TRANSFORM_TRAIN,
        key_map=KEY_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        map_class_id_to_class_name=map_class_id_to_class_name,
    )
    data_module.setup("fit")
    assert data_module.train_dataset is not None, "train_dataset 应该被初始化"
    assert data_module.val_dataset is not None, "val_dataset 应该被初始化"

    data_module.setup("validate")
    assert data_module.val_dataset is not None, "val_dataset 应该被初始化"

    data_module.setup("test")
    assert data_module.test_dataset is not None, "test_dataset 应该被初始化"

    data_module.setup("predict")
    assert data_module.pred_dataset is not None, "predict_dataset 应该被初始化"


def test_ImageClassifierDataModule_train_dataloader():
    data_module = ImageClassifierDataModule(
        train_csv_paths=PATH_TRAIN_CSV_PATHS,
        val_csv_paths=PATH_TRAIN_CSV_PATHS,
        test_csv_paths=PATH_TRAIN_CSV_PATHS,
        predict_csv_paths=PATH_PREDICT_CSV_PATHS,
        transform_train=TRANSFORM_TRAIN,
        transform_val=TRANSFORM_TRAIN,
        transform_test=TRANSFORM_TRAIN,
        transform_predict=TRANSFORM_TRAIN,
        key_map=KEY_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        map_class_id_to_class_name=map_class_id_to_class_name,
    )
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    net_in, net_out = next(iter(train_dataloader))

    assert ("img_tv_transformed" in net_in.keys()) == True
    assert ("class_id" in net_out.keys()) == True

    assert net_in["img_tv_transformed"].shape == (BATCH_SIZE, 3, 224, 224)
    assert net_out["class_id"].shape == (BATCH_SIZE,)


def test_ImageClassifierDataModule_predict_dataloader():
    data_module = ImageClassifierDataModule(
        train_csv_paths=PATH_TRAIN_CSV_PATHS,
        val_csv_paths=PATH_TRAIN_CSV_PATHS,
        test_csv_paths=PATH_TRAIN_CSV_PATHS,
        predict_csv_paths=PATH_PREDICT_CSV_PATHS,
        transform_train=TRANSFORM_TRAIN,
        transform_val=TRANSFORM_TRAIN,
        transform_test=TRANSFORM_TRAIN,
        transform_predict=TRANSFORM_TRAIN,
        key_map=KEY_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        map_class_id_to_class_name=map_class_id_to_class_name,
    )
    data_module.setup("predict")
    pred_dataloader = data_module.predict_dataloader()
    net_in, net_out = next(iter(pred_dataloader))

    assert ("img_tv_transformed" in net_in.keys()) == True
    assert net_in["img_tv_transformed"].shape == (BATCH_SIZE, 3, 224, 224)
    assert net_out == {}
