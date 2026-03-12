import torch
import lightning.pytorch as pl
import yaml
from torchvision import transforms

from lovely_deep_learning.data_module.image_nette import ImageNetteDataModule
from lovely_deep_learning.module.image_classifier import ImageClassifierModule
from lovely_deep_learning.callback.image_classifier_save_prediction import SavePredictionCallback




def test_ImageClassifierModule_train():
    path_yaml = "configs/experiments/image_classifiter.yaml"
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ImageClassifierModule(**config["model"]["init_args"])
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dm = ImageNetteDataModule(
        data_dir="/home/xiaopangdun/project/deep_learning/src/train/datasets/IMAGENETTE/imagenette2-320",
        batch_size=1,
        transform_train=train_transform,
        transform_val=val_transform,
        num_workers=1,
    )

    trainer = pl.Trainer(max_epochs=5, fast_dev_run=True)
    trainer.fit(model, datamodule=dm)


def test_ImageClassifierModule_predict():
    path_yaml = "configs/experiments/image_classifiter.yaml"
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = ImageClassifierModule(**config["model"]["init_args"])
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dm = ImageNetteDataModule(
        data_dir="/home/xiaopangdun/project/deep_learning/src/train/datasets/IMAGENETTE/imagenette2-320",
        batch_size=1,
        transform_train=train_transform,
        transform_val=val_transform,
        num_workers=1,
    )
    callback = SavePredictionCallback(save_dir="/home/xiaopangdun/project/deep_learning/src/train/tmp")
    trainer = pl.Trainer(callbacks=[callback])
    trainer.predict(model, datamodule=dm)
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
