from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from .base import BaseDataModule
from ..dataset.box_crop_distribution import format_distribution_report, training_class_rows
from ..dataset.image_classifier import ImageClassifierDataset, ImageClassifierFromDetectBoxDataset


class ImageClassifierDataModule(BaseDataModule):
    """为 ``fit`` / ``validate`` / ``test`` / ``predict`` 各阶段挂载 ``ImageClassifierDataset``。

    - 训练、验证、测试通常共用 ``key_map``（与带标签 CSV 表头一致）。
    - 预测 CSV 往往仅有图像路径列，应通过 ``predict_key_map`` 传入与预测表头匹配的映射。
      ``key_map`` 中的列名必须都在表头里，预测表不能沿用带标签的默认映射。
    - ``norm_mean`` / ``norm_std`` 传给 Dataset，供 ``convert_img_from_tensor_to_numpy`` 等反标准化
      可视化路径使用。
    - ``class_groups`` 决定训练 id。不传则 CSV 里出现的类别全部训练。
      绘图名称从有类别列的 CSV 推导。预测表没有类别列时，用同一次 ``setup`` 里训练集推导出的那份；
      单独 ``predict``、训练集尚未建立时，预测图上的名字是数字 id。
    - 框级分类用同文件的 ``ImageClassifierFromDetectBoxDataModule``，数据集换成
      ``ImageClassifierFromDetectBoxDataset``。
    """

    def __init__(
        self,
        key_map=None,
        predict_key_map=None,
        norm_mean=None,
        norm_std=None,
        class_groups=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        key_map
            训练 / 验证 / 测试 CSV 的列名映射；``None`` 时使用 ``ImageClassifierDataset`` 内置默认。
        predict_key_map
            预测 CSV 的列名映射；常比 ``key_map`` 少标签相关项。
        norm_mean, norm_std
            与训练时 Normalize 一致的均值、方差，用于 tensor→numpy 可视化时的反标准化。
        class_groups
            ``{训练 id: [原始 class_id, ...]}``，交给各有标签的 Dataset。
            值为 ``background`` 时只有框级数据集能用。``None`` 表示全部训练。
        **kwargs
            交给 ``BaseDataModule.__init__``（各阶段 ``*_csv_paths``、``transform_*``、
            ``batch_size``、``num_workers`` 等）。
        """
        super().__init__(**kwargs)
        self.key_map = key_map
        self.predict_key_map = predict_key_map
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.class_groups = class_groups

    def setup(self, stage=None):
        """按 Lightning ``stage`` 创建对应 Dataset；``stage is None`` 时构建全部阶段所用数据集。

        ``fit`` 与 ``validate`` 分支均会构造 ``val_dataset``（前者在完整训练流程中一并准备验证集；
        单独 ``trainer.validate`` 时走 ``validate``）。``stage is None``（如部分 CLI 流程）下各 ``if``
        均可能执行，验证集会被构造两次，结果等价。
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_classifier_dataset(
                self.train_csv_paths,
                self.key_map,
                self.transform_train,
                for_train=True,
                map_class_id_to_class_name=None,
            )
            self.val_dataset = self._make_classifier_dataset(
                self.val_csv_paths,
                self.key_map,
                self.transform_val,
                map_class_id_to_class_name=None,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self._make_classifier_dataset(
                self.val_csv_paths,
                self.key_map,
                self.transform_val,
                map_class_id_to_class_name=None,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._make_classifier_dataset(
                self.test_csv_paths,
                self.key_map,
                self.transform_test,
                map_class_id_to_class_name=None,
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = self._make_classifier_dataset(
                self.predict_csv_paths,
                self.predict_key_map,
                self.transform_predict,
                map_class_id_to_class_name=self._class_names_for_predict(),
            )

    def _make_classifier_dataset(
        self, csv_paths, key_map, transform, *, for_train: bool = False, map_class_id_to_class_name=None
    ):
        """整图分类忽略 ``for_train``。框级子类只用它给训练集追加背景样本。"""
        _ = for_train
        return ImageClassifierDataset(
            csv_paths,
            key_map=key_map,
            transform=transform,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            class_groups=self.class_groups,
            map_class_id_to_class_name=map_class_id_to_class_name,
        )

    def _class_names_for_predict(self) -> Optional[dict[int, str]]:
        """预测表没有类别列时的绘图名称。只使用同一次 ``setup`` 里训练集推导出的那份。"""
        train = self.train_dataset
        if train is not None and train.map_class_id_to_class_name:
            return dict(train.map_class_id_to_class_name)
        return None

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.pred_dataset.get_collate_fn_for_dataloader(),
        )


class ImageClassifierFromDetectBoxDataModule(ImageClassifierDataModule):
    """框级分类。CSV、transform、``key_map`` 与整图分类相同，额外传入裁剪和背景参数。

    模型、损失、指标、后处理仍用 ``ImageClassifierModule``。
    ``prepare_data`` 不生成 CSV，先运行 ``scripts/yolo_to_box_crop_csv.py``。
    ``class_groups`` 与整图分类相同，决定训练 id。值为 ``background`` 的项没有 CSV 行。
    ``background_ratio`` 只加在训练集上；验证、测试和预测为 0。

    ``fit`` 时为训练集和验证集各打一张类别表（筛完并完成类融合之后），并写入
    ``trainer.log_dir/data_distribution.txt``。
    """

    def __init__(
        self,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
        background_ratio: float = 0.0,
        background_iou: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.box_scale = float(box_scale)
        self.min_side_px = float(min_side_px)
        self.background_ratio = float(background_ratio)
        self.background_iou = float(background_iou)

    def _make_classifier_dataset(
        self, csv_paths, key_map, transform, *, for_train: bool = False, map_class_id_to_class_name=None
    ):
        return ImageClassifierFromDetectBoxDataset(
            csv_paths,
            key_map=key_map,
            transform=transform,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            box_scale=self.box_scale,
            min_side_px=self.min_side_px,
            class_groups=self.class_groups,
            background_ratio=self.background_ratio if for_train else 0.0,
            background_iou=self.background_iou,
            map_class_id_to_class_name=map_class_id_to_class_name,
        )

    def setup(self, stage=None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self._report_fit_distribution()

    def _report_fit_distribution(self) -> None:
        named = (
            ("train", "训练集", self.train_csv_paths, self.train_dataset),
            ("val", "验证集", self.val_csv_paths, self.val_dataset),
        )
        splits = []
        for split, title, csv_paths, dataset in named:
            record = {
                "split": split,
                "title": title,
                "csv_paths": [str(path) for path in csv_paths],
                "classes": training_class_rows(dataset.distribution["classes"]),
            }
            warnings = dataset.distribution["mapping"]["warnings"]
            if warnings:
                record["warnings"] = warnings
            splits.append(record)
        payload = {"splits": splits}
        self.data_distribution = payload
        text = format_distribution_report(payload)
        trainer = self.trainer
        log_dir = None if trainer is None else trainer.log_dir
        if trainer is not None and not getattr(trainer, "is_global_zero", True):
            return
        print(text)
        if trainer is None or not log_dir:
            return
        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data_distribution.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"数据分布已写入 {path}")
