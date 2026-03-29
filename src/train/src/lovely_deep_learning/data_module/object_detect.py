from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseDataModule
from ..dataset.image_classifier import ImageClassifierDataset
from ..dataset.object_detect import ObjectDetectDataset


class ObjectDetectDataModule(BaseDataModule):
    """为 ``fit`` / ``validate`` / ``test`` / ``predict`` 各阶段挂载 ``ObjectDetectDataset``。

    - 训练、验证、测试通常共用 ``key_map``（与带标签 CSV 表头一致，须含图像与 YOLO 标签路径列）。
    - 预测 CSV 往往仅有图像路径列，应通过 ``predict_key_map`` 传入与预测表头匹配的映射；
      若为 ``None``，须保证 ``ObjectDetectDataset`` 默认 ``key_map`` 与 CSV 表头可对应。
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
            训练 / 验证 / 测试 CSV 的列名映射；``None`` 时使用 ``ObjectDetectDataset`` 内置默认。
        predict_key_map
            预测 CSV 的列名映射；常比 ``key_map`` 少 ``object_label_path`` 等项。
        map_class_id_to_class_name
            ``None`` / ``dict`` / 指向 CSV 的 ``str``（表头须含 ``class_id``、``class_name``）。
            原始值保存在 ``_map_class_id_to_class_name_spec``；在 ``setup`` 中解析为 ``dict`` 并写入
            ``map_class_id_to_class_name`` 后传给各 Dataset（``str`` 路径须已存在，通常由子类
            ``prepare_data`` 生成）。构造后、``setup`` 前请勿依赖 ``map_class_id_to_class_name``。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，用于 tensor→numpy 可视化时的反标准化。
        **kwargs
            交给 ``BaseDataModule.__init__``（各阶段 ``*_csv_paths``、``transform_*``、
            ``batch_size``、``num_workers`` 等）。
        """
        super().__init__(**kwargs)
        self.key_map = key_map
        self.predict_key_map = predict_key_map
        self._map_class_id_to_class_name_spec = map_class_id_to_class_name
        self.map_class_id_to_class_name: dict[int, str] = {}
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def setup(self, stage=None):
        """按 Lightning ``stage`` 创建对应 Dataset；``stage is None`` 时构建全部阶段所用数据集。

        ``fit`` 与 ``validate`` 分支均会构造 ``val_dataset``（前者在完整训练流程中一并准备验证集；
        单独 ``trainer.validate`` 时走 ``validate``）。``stage is None``（如部分 CLI 流程）下各 ``if``
        均可能执行，验证集会被构造两次，结果等价。
        """
        self.map_class_id_to_class_name = self._resolve_map_class_id_spec(
            self._map_class_id_to_class_name_spec
        )
        if stage == "fit" or stage is None:
            self.train_dataset = ObjectDetectDataset(
                self.train_csv_paths,
                key_map=self.key_map,
                transform=self.transform_train,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
            self.val_dataset = ObjectDetectDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = ObjectDetectDataset(
                self.val_csv_paths,
                key_map=self.key_map,
                transform=self.transform_val,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ObjectDetectDataset(
                self.test_csv_paths,
                key_map=self.key_map,
                transform=self.transform_test,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = ObjectDetectDataset(
                self.predict_csv_paths,
                self.predict_key_map,
                transform=self.transform_predict,
                map_class_id_to_class_name=self.map_class_id_to_class_name,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )

    @staticmethod
    def _resolve_map_class_id_spec(spec: Any) -> dict[int, str]:
        """``setup`` 使用：``None`` / ``dict`` / 已存在的 CSV 路径 → id→类别名；路径不存在则报错。"""
        if spec is None:
            return {}
        if isinstance(spec, dict):
            return {int(k): str(v) for k, v in spec.items()}
        if isinstance(spec, str):
            p = Path(spec).expanduser()
            if not p.is_file():
                raise FileNotFoundError(
                    "map_class_id_to_class_name 指定的 CSV 不存在（须先成功执行 prepare_data 或修正路径）："
                    f" {p}"
                )
            return ImageClassifierDataset.load_map_class_id_to_class_name_from_csv(p)
        raise TypeError(
            "map_class_id_to_class_name 须为 None、dict 或 str（CSV 路径），"
            f"收到 {type(spec)!r}"
        )

    @staticmethod
    def _map_spec_effective_for_csv_generation(spec: Any) -> dict[int, str]:
        """仅 ``prepare_data`` 内生成 CSV 时用：路径上文件尚未写出时当作无映射 ``{}``。"""
        if spec is None:
            return {}
        if isinstance(spec, dict):
            return {int(k): str(v) for k, v in spec.items()}
        if isinstance(spec, str):
            p = Path(spec).expanduser()
            if p.is_file():
                return ImageClassifierDataset.load_map_class_id_to_class_name_from_csv(p)
            return {}
        raise TypeError(
            "map_class_id_to_class_name 须为 None、dict 或 str（CSV 路径），"
            f"收到 {type(spec)!r}"
        )

    @staticmethod
    def _write_map_class_id_to_class_name_csv(
        path: Path, class_name_to_idx: dict[str, int]
    ) -> None:
        """将 ``{class_name: class_id}`` 写成表头为 ``class_id``, ``class_name`` 的 CSV。"""
        rows = sorted((idx, name) for name, idx in class_name_to_idx.items())
        pd.DataFrame(rows, columns=["class_id", "class_name"]).to_csv(
            path, index=False)
