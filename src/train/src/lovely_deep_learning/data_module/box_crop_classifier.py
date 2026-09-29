"""框级分类 DataModule。转换由脚本完成，这里只把 CSV 交给 ``BoxCropClassifierDataset``。"""

from pathlib import Path
from typing import Any, Optional, Sequence

from .image_classifier import ImageClassifierDataModule
from ..dataset.box_crop_classifier import BoxCropClassifierDataset
from ..dataset.box_crop_distribution import format_distribution_report, training_class_rows


class BoxCropClassifierDataModule(ImageClassifierDataModule):
    """与图像分类 DataModule 的 CSV、transform、``key_map`` 相同，额外传入 ``box_scale``。

    模型、损失、指标、后处理仍用 ``ImageClassifierModule`` 那一套。
    ``prepare_data`` 不生成 CSV，先运行 ``scripts/yolo_to_box_crop_csv.py``。
    ``class_groups`` 为 ``{训练 id: [原始 class_id, ...]}`` 时，Dataset 在读入时合并类别。

    ``fit`` 时为训练集和验证集各打一张类别表（筛完并完成类融合之后），并写入
    ``trainer.log_dir/data_distribution.txt``。
    """

    def __init__(
        self,
        box_scale: float = 1.2,
        min_side_px: float = 16.0,
        class_groups: Optional[dict[Any, Sequence[int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.box_scale = float(box_scale)
        self.min_side_px = float(min_side_px)
        self.class_groups = class_groups

    def _make_dataset(self, csv_paths, key_map, transform):
        return BoxCropClassifierDataset(
            csv_paths,
            key_map=key_map,
            transform=transform,
            map_class_id_to_class_name=self.map_class_id_to_class_name,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            box_scale=self.box_scale,
            min_side_px=self.min_side_px,
            class_groups=self.class_groups,
        )

    def setup(self, stage=None):
        self.map_class_id_to_class_name = self._resolve_map_class_id_spec(
            self._map_class_id_to_class_name_spec
        )
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(
                self.train_csv_paths, self.key_map, self.transform_train
            )
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self._make_dataset(
                self.val_csv_paths, self.key_map, self.transform_val
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._make_dataset(
                self.test_csv_paths, self.key_map, self.transform_test
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = self._make_dataset(
                self.predict_csv_paths, self.predict_key_map, self.transform_predict
            )
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
