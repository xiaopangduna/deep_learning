from torchvision.transforms import v2
import torch

from lovely_deep_learning.data_module.object_detect import ObjectDetectDataModule

PATH_TRAIN_CSV_PATHS = ["tests/test_data/coco8/train.csv"]
PATH_PREDICT_CSV_PATHS = ["tests/test_data/coco8/predict.csv"]

KEY_MAP = {"img_path": "path_img", "object_label_path": "path_label_detect_yolo"}
PREDICT_KEY_MAP = {"img_path": "path_img"}
BATCH_SIZE = 1
NUM_WORKERS = 0
TRANSFORM_TRAIN = v2.Compose(
    [v2.Resize(size=(640, 640)), v2.ToDtype(dtype=torch.float32, scale=True)]
)

map_class_id_to_class_name = {
    0: "person",
    1: "bicycle",
}


def test_ObjectDetectDataModule_init():

    data_module = ObjectDetectDataModule(
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

    assert data_module.map_class_id_to_class_name == {}
    data_module.setup("fit")
    assert data_module.map_class_id_to_class_name == map_class_id_to_class_name


def test_ObjectDetectDataModule_setup():
    data_module = ObjectDetectDataModule(
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


def test_ObjectDetectDataModule_train_dataloader():
    data_module = ObjectDetectDataModule(
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

    assert "img" in net_in[0] and net_in[0]["img"].shape[0] == 3
    assert "img_tv_transformed" in net_in[0].keys()
    assert "cls_np" in net_out[0].keys()

    assert net_in[0]["img_tv_transformed"].shape == (3, 640, 640)


def test_ObjectDetectDataModule_predict_dataloader():
    data_module = ObjectDetectDataModule(
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

    assert net_in[0]["img"].shape == (3, 640, 640)
    assert "img_tv_transformed" in net_in[0].keys()
    assert net_in[0]["img_tv_transformed"].shape == (3, 640, 640)
    assert net_out[0] == {}
