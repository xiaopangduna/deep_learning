import io
from pathlib import Path
from zipfile import ZipFile
from typing import Union
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader, random_split


class COCO8DataModule(pl.LightningDataModule):
    COCO8_URL = (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
    )

    def __init__(
        self,
        data_dir: Union[str, Path] = "./datasets",
        batch_size: int = 4,
        num_workers: int = 2,
        transform=None,
        download: bool = False,
        download_url=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.download = download
        self.download_url = download_url or self.COCO8_URL

    def prepare_data(self):
        # 定义 train 和 val 路径
        train_path = self.data_dir / "images" / "train"
        val_path = self.data_dir / "images" / "val"

        # 简单检查路径是否存在
        if train_path.exists() and val_path.exists():
            print(f"✅ COCO8 数据集已存在: {self.data_dir}")
            return

        # 下载逻辑
        if self.download:
            print(f"⬇️ COCO8 数据集未找到，开始下载...")
            self.data_dir.mkdir(parents=True, exist_ok=True)

            zip_path = self.data_dir / "coco8.zip"
            # 下载到本地 zip 文件
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 解压 zip
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            # 删除 zip 文件
            zip_path.unlink()
            print(f"✅ 下载并解压完成，已删除 zip 文件: {self.data_dir}")
        else:
            raise RuntimeError(
                f"COCO8 数据集未找到，请手动从https://docs.ultralytics.com/zh/datasets/detect/coco8/#dataset-yaml下载到 {self.data_dir}"
            )

    def setup(self, stage=None):

        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):

        pass

    def predict_dataloader(self):

        pass

    def teardown(self, stage=None):
        # 清理操作（可选）
        pass
