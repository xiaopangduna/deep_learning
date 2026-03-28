from .base import BaseDataModule
from ..dataset.image_classifier import ImageClassifierDataset


class ImageClassifierDataModule(BaseDataModule):
    """为 ``fit`` / ``validate`` / ``test`` / ``predict`` 各阶段挂载 ``ImageClassifierDataset``。

    - 训练、验证、测试通常共用 ``key_map``（与带标签 CSV 表头一致）。
    - 预测 CSV 往往仅有图像路径列，应通过 ``predict_key_map`` 传入与预测表头匹配的映射；
      若为 ``None``，须保证传给 ``ImageClassifierDataset`` 的默认 ``key_map`` 仍能与 CSV 表头求交
      （参见 ``ImageClassifierDataset._key_map_intersect_csv_headers``）。
    - ``norm_mean`` / ``norm_std`` 传给 Dataset，供 ``convert_img_from_tensor_to_numpy`` 等反标准化
      可视化路径使用。
    """

    def __init__(
        self,
        key_map=None,
        predict_key_map=None,
        map_class_id_to_class_name=None,
        norm_mean=None,
        norm_std=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        key_map
            训练 / 验证 / 测试 CSV 的列名映射；``None`` 时使用 ``ImageClassifierDataset`` 内置默认。
        predict_key_map
            预测 CSV 的列名映射；常比 ``key_map`` 少标签相关项。
        map_class_id_to_class_name
            显式 id→类别名表；``None`` 且表中含 ``class_id``/``class_name`` 时由 Dataset 从表推断。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，用于 tensor→numpy 可视化时的反标准化。
        **kwargs
            交给 ``BaseDataModule.__init__``（各阶段 ``*_csv_paths``、``transform_*``、
            ``batch_size``、``num_workers`` 等）。
        """
        super().__init__(**kwargs)
        self.key_map = key_map
        self.predict_key_map = predict_key_map
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def setup(self, stage=None):
        """按 Lightning ``stage`` 创建对应 Dataset；``stage is None`` 时构建全部阶段所用数据集。

        ``fit`` 与 ``validate`` 分支均会构造 ``val_dataset``（前者在完整训练流程中一并准备验证集；
        单独 ``trainer.validate`` 时走 ``validate``）。``stage is None``（如部分 CLI 流程）下各 ``if``
        均可能执行，验证集会被构造两次，结果等价。
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ImageClassifierDataset(
                self.train_csv_paths,
                key_map=self.key_map,
                transform=self.transform_train,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = ImageClassifierDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ImageClassifierDataset(
                self.test_csv_paths,
                key_map=self.key_map,
                transform=self.transform_test,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = ImageClassifierDataset(
                self.predict_csv_paths,
                self.predict_key_map,
                transform=self.transform_predict,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
