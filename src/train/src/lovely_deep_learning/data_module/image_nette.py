import os
from typing import Optional
from pathlib import Path
import urllib.request
import tarfile
import pandas as pd
import lightning.pytorch as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lovely_deep_learning.dataset.predict import ImagePredictDataset
from .base import BaseDataModule
from .image_classifier import ImageClassifierDataModule


class ImageNetteDataModule(ImageClassifierDataModule):
    def __init__(self, dataset_dir="datasets/IMAGENETTE/imagenette2-320", map_class_id_to_class_name={}, **kwargs):
        super().__init__(**kwargs)
        self.dataset_dir = Path(dataset_dir)
        self.map_class_id_to_class_name = map_class_id_to_class_name

    def _generate_csv_files(self, dataset_dir: Path):
        """
        生成三个CSV文件：train.csv、val.csv和predict.csv
        Args:
            dataset_dir: 包含图像数据的目录路径
        """
        dataset_dir = Path(dataset_dir)
        output_dir = dataset_dir.parent  # 修改输出目录路径
        output_dir.mkdir(exist_ok=True)

        # 检查文件是否存在，如果存在则跳过生成
        train_csv_path = output_dir / "train.csv"
        val_csv_path = output_dir / "val.csv"
        predict_csv_path = output_dir / "predict.csv"

        if train_csv_path.exists() and val_csv_path.exists() and predict_csv_path.exists():
            print("============================================================================")
            print(f"CSV files already exist, skipping generation:")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {predict_csv_path}")
            print("============================================================================")
            return

        # 获取类别名称到ID的映射
        # 如果提供了map_class_id_to_class_name，则使用它来构建class_to_idx映射
        if self.map_class_id_to_class_name:
            # 从map_class_id_to_class_name反向构建class_to_idx
            class_to_idx = {name: idx for idx, name in self.map_class_id_to_class_name.items()}
        else:
            # 否则按原有方式自动构建
            classes = sorted([d.name for d in (dataset_dir / "train").iterdir() if d.is_dir()])
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # 生成训练集CSV
        train_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "train" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        train_rows.append({
                            'path_img': f"imagenette2-320/train/{class_name}/{img_path.name}",
                            'class_name': class_name,
                            'class_id': class_to_idx[class_name]
                        })
        
        train_df = pd.DataFrame(train_rows, columns=['path_img', 'class_name', 'class_id'])
        train_df.to_csv(train_csv_path, index=False)

        # 生成验证集CSV
        val_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "val" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        val_rows.append({
                            'path_img': f"imagenette2-320/val/{class_name}/{img_path.name}",
                            'class_name': class_name,
                            'class_id': class_to_idx[class_name]
                        })

        val_df = pd.DataFrame(val_rows, columns=['path_img', 'class_name', 'class_id'])
        val_df.to_csv(val_csv_path, index=False)

        # 生成预测集CSV（通常是从验证集中选择的图像，不含标签信息）
        predict_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "val" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        predict_rows.append({
                            'path_img': f"imagenette2-320/val/{class_name}/{img_path.name}"
                        })

        predict_df = pd.DataFrame(predict_rows, columns=['path_img'])
        predict_df.to_csv(predict_csv_path, index=False)
        
        print("============================================================================")
        print(f"Generated {str(train_csv_path)} with {len(train_df)} entries")
        print(f"Generated {str(val_csv_path)} with {len(val_df)} entries")
        print(f"Generated {str(predict_csv_path)} with {len(predict_rows)} entries")
        print("============================================================================")

    def prepare_data(self):
        """
        下载并解压 ImageNette 320px版本。
        self.data_dir 指向解压后的目录
        """
        parent_dir = self.dataset_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        imagenette_tar = parent_dir / "imagenette2-320.tgz"

        if not self.dataset_dir.exists():
            print(f"Downloading ImageNette dataset to {imagenette_tar} ...")
            urllib.request.urlretrieve(imagenette_url, str(imagenette_tar))

            print(f"Extracting {imagenette_tar} ...")
            with tarfile.open(imagenette_tar, "r:gz") as tar:
                tar.extractall(path=str(parent_dir))  # 解压到父目录

            print("Download and extraction complete.")
        else:
            print("ImageNette dataset already exists, skipping download.")

        # 生成CSV文件
        self._generate_csv_files(self.dataset_dir)
