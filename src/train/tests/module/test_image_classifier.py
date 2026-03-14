import torch
import lightning.pytorch as pl
import yaml
from torchvision import transforms
from torchvision.transforms import v2
from lovely_deep_learning.data_module.image_nette import ImageNetteDataModule
from lovely_deep_learning.module.image_classifier import ImageClassifierModule
from lovely_deep_learning.callback.image_classifier import ImageClassifierCallback
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


def test_ImageClassifierModule_train():
    path_yaml = "configs/experiments/image_classifiter_copy.yaml"
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ImageClassifierModule(**config["model"]["init_args"])

    dm = ImageClassifierDataModule(
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
    )
    callback = ImageClassifierCallback(save_dir="tmp")
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[callback])

    trainer.fit(model, datamodule=dm)
    pass


def test_ImageClassifierModule_test():
    path_yaml = "configs/experiments/image_classifiter_copy.yaml"
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ImageClassifierModule(**config["model"]["init_args"])

    dm = ImageClassifierDataModule(
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

    callback = ImageClassifierCallback(save_dir="./tmp")
    trainer = pl.Trainer(callbacks=[callback],limit_test_batches=5)
    trainer.test(model, datamodule=dm)
    pass


def test_ImageClassifierModule_predict():
    path_yaml = "configs/experiments/image_classifiter_copy.yaml"
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ImageClassifierModule(**config["model"]["init_args"])

    dm = ImageClassifierDataModule(
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

    callback = ImageClassifierCallback(save_dir="./tmp")
    trainer = pl.Trainer(callbacks=[callback])
    trainer.predict(model, datamodule=dm,limit_predict_batches=5)
    pass


if __name__ == "__main__":

    ckpt_path = "/home/xiaopangdun/project/deep_learning/src/train/logs/image_classifiter/version_2/checkpoints/epoch=1-step=1184.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 数据模块
    dm = ImageNetteDataModule(
        data_dir="/home/xiaopangdun/project/deep_learning/src/train/datasets/IMAGENETTE/imagenette2-320", batch_size=4
    )
    # 模型
    model = ImageClassifierModule(num_classes=10, learning_rate=1e-3)

    # Trainer
    trainer = pl.Trainer(max_epochs=5)
    # trainer.fit(model, datamodule=dm)

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    pass
