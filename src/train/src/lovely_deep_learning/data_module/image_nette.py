import os
from pathlib import Path
import urllib.request
import tarfile
import lightning.pytorch as pl
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

class ImageNetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./datasets/IMAGENETTE/imagenette2-320", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)  # 指向 train/val 所在目录
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        下载并解压 ImageNette 320px版本。
        self.data_dir 指向解压后的目录
        """
        parent_dir = self.data_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        imagenette_tar = parent_dir / "imagenette2-320.tgz"

        if not self.data_dir.exists():
            print(f"Downloading ImageNette dataset to {imagenette_tar} ...")
            urllib.request.urlretrieve(imagenette_url, str(imagenette_tar))

            print(f"Extracting {imagenette_tar} ...")
            with tarfile.open(imagenette_tar, "r:gz") as tar:
                tar.extractall(path=str(parent_dir))  # 解压到父目录

            print("Download and extraction complete.")
        else:
            print("ImageNette dataset already exists, skipping download.")

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        self.train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
        self.val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)