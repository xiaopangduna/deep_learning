import torch
import lightning.pytorch as pl

from lovely_deep_learning.data_module.image_nette import ImageNetteDataModule
from lovely_deep_learning.module.image_classifier import ImageClassifierModule


if __name__ == "__main__":

    ckpt_path = "/home/xiaopangdun/project/deep_learning/src/train/logs/mnist/version_1/checkpoints/epoch=0-step=592.ckpt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # 数据模块
    dm = ImageNetteDataModule(data_dir="/home/xiaopangdun/project/deep_learning/src/train/datasets/IMAGENETTE/imagenette2-320", batch_size=4)
    # 模型
    model = ImageClassifierModule(num_classes=10, lr=1e-3)

    # Trainer
    trainer = pl.Trainer(max_epochs=5)  
    trainer.fit(model, datamodule=dm)
    pass