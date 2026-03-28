from typing import List

import lightning.pytorch as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    """封装训练 / 验证 / 测试 / 预测四段 CSV 与对应 ``DataLoader`` 的默认行为。

    各阶段使用的 ``torch.utils.data.Dataset`` 由子类在 ``setup(stage)`` 里赋值；本类仅负责
    用统一的 ``batch_size``、``num_workers`` 以及 dataset 提供的 ``collate_fn`` 构建
    ``DataLoader``（预测 DataLoader 例外，见 ``predict_dataloader``）。

    Attributes
    ----------
    train_csv_paths, val_csv_paths, test_csv_paths, predict_csv_paths : List[str]
        各阶段 CSV 路径列表，具体语义由子类 Dataset 解析。
    transform_train, transform_val
        训练与验证用的样本变换（通常为 ``Compose`` 或可调用对象）。
    transform_test, transform_predict
        未传时分别回退为 ``transform_val``。
    batch_size, num_workers
        各 ``DataLoader`` 的批量大小与 ``DataLoader`` worker 数。
    train_dataset, val_dataset, test_dataset, pred_dataset
        初始为 ``None``，须在子类 ``setup`` 中赋值后再调用对应 ``*_dataloader``。
    """

    def __init__(
        self,
        train_csv_paths: List[str],
        val_csv_paths: List[str],
        test_csv_paths: List[str],
        predict_csv_paths: List[str],
        transform_train,
        transform_val,
        transform_test=None,
        transform_predict=None,
        batch_size=32,
        num_workers=4,
    ):
        """
        Parameters
        ----------
        train_csv_paths, val_csv_paths, test_csv_paths, predict_csv_paths
            各 Lightning stage 使用的 CSV 路径列表。
        transform_train, transform_val
            训练、验证集变换。
        transform_test, transform_predict
            可选；默认沿用 ``transform_val``。
        batch_size
            所有阶段 DataLoader 的 batch 大小。
        num_workers
            ``DataLoader`` 后台加载进程数。
        """
        super().__init__()
        self.train_csv_paths = train_csv_paths
        self.val_csv_paths = val_csv_paths
        self.test_csv_paths = test_csv_paths
        self.predict_csv_paths = predict_csv_paths
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test or transform_val
        self.transform_predict = transform_predict or transform_val
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

    def train_dataloader(self):
        """打乱顺序的训练 DataLoader，使用 ``train_dataset.get_collate_fn_for_dataloader()``。"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.get_collate_fn_for_dataloader(),
        )

    def val_dataloader(self):
        """验证 DataLoader；当前实现中 ``shuffle=True``（与常见验证集不打乱不同，需注意）。"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.get_collate_fn_for_dataloader(),
        )

    def test_dataloader(self):
        """测试 DataLoader，不打乱顺序。"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.get_collate_fn_for_dataloader(),
        )

    def predict_dataloader(self):
        """预测 DataLoader；未设置 ``collate_fn``，依赖 PyTorch 默认拼接行为。"""
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
