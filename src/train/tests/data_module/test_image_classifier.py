import pytest

from torchvision.transforms import v2
import torch

from lovely_deep_learning.data_module.image_classifier import ImageClassifierDataModule

PATH_TRAIN_CSV_PATHS = ["tests/test_data/dataset/test_image_classifier_train.csv"]
PATH_PREDICT_CSV_PATHS = ["tests/test_data/dataset/test_image_classifier_predict.csv"]

KEY_MAP = {"img_path": "path_img", "class_name": "class_name", "class_id": "class_id"}
PREDICT_KEY_MAP = {"img_path": "path_img"}
BATCH_SIZE = 1
NUM_WORKERS = 1
TRANSFORM_TRAIN = v2.Compose([v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)])

map_class_id_to_class_name = {
    0: "n01440764",
    1: "n02102040",
}


def test_ImageClassifierDataModule_init():

    data_module = ImageClassifierDataModule(
        train_csv_paths=PATH_TRAIN_CSV_PATHS,
        val_csv_paths=PATH_TRAIN_CSV_PATHS,
        test_csv_paths=PATH_TRAIN_CSV_PATHS,
        predict_csv_paths=PATH_PREDICT_CSV_PATHS,
        transform_train=TRANSFORM_TRAIN,
        transform_val=TRANSFORM_TRAIN,
        transform_test=TRANSFORM_TRAIN,
        transform_predict=TRANSFORM_TRAIN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        
        key_map=KEY_MAP,
        predict_key_map=PREDICT_KEY_MAP,
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
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        key_map=KEY_MAP,
        predict_key_map=PREDICT_KEY_MAP,
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
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        key_map=KEY_MAP,
        predict_key_map=PREDICT_KEY_MAP,
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
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        key_map=KEY_MAP,
        predict_key_map=PREDICT_KEY_MAP,
        map_class_id_to_class_name=map_class_id_to_class_name,
    )
    data_module.setup("predict")
    pred_dataloader = data_module.predict_dataloader()
    net_in, net_out = next(iter(pred_dataloader))

    assert ("img_tv_transformed" in net_in.keys()) == True
    assert net_in["img_tv_transformed"].shape == (BATCH_SIZE, 3, 224, 224)
    assert net_out == {}
