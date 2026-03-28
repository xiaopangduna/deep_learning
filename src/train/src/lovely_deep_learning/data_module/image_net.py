import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import pandas as pd

from .image_classifier import ImageClassifierDataModule


class ImageNetDataModule(ImageClassifierDataModule):
    """ImageNet ILSVRC CLS-LOC：不下载数据，在 ``prepare_data`` 中生成 train/val/test/predict 的 CSV，
    并沿用图像分类 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. ``trainer.fit`` 之前 Lightning 会调用 ``prepare_data``：在 ``dataset_dir`` 下生成 train/val/test/predict 四个 CSV
       及 ``map_class_id_to_class_name.csv``（若主 CSV 已存在则跳过整批生成；仅缺映射文件时会补写映射）。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ImageClassifierDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/IMAGENET``，即 ILSVRC 布局根目录，须包含：
      ``ILSVRC/Data/CLS-LOC/train/<synset>/*.JPEG``、``.../val/*.JPEG``、``.../test/*.JPEG``、
      ``ILSVRC/Annotations/CLS-LOC/val/*.xml``、
      ``ILSVRC/ImageSets/CLS-LOC/train_cls.txt``、``val.txt``、``test.txt``。
    - 生成的 CSV 写在 ``dataset_dir`` 下；``path_img`` 为相对 ``dataset_dir`` 的路径，与 ``BaseDataset`` 按 CSV 目录解析一致。
    - ``train.csv`` / ``val.csv``：来自 ``train_cls`` 与 val XML 标签；``test.csv`` / ``predict.csv``：仅 ``path_img``，来自 ``test.txt`` 列表。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ImageClassifierDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。若提供 ``map_class_id_to_class_name``（``dict`` 或 CSV 路径字符串），
    生成 CSV 时的 synset→id 以该映射为准；否则按 train 下 synset 文件夹名字典序分配 id。
    """

    _REL_DATA = Path("ILSVRC/Data/CLS-LOC")
    _REL_ANN = Path("ILSVRC/Annotations/CLS-LOC")
    _REL_SETS = Path("ILSVRC/ImageSets/CLS-LOC")

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
        batch_size: int = 32,
        num_workers: int = 4,
        key_map=None,
        predict_key_map=None,
        map_class_id_to_class_name=None,
        norm_mean=None,
        norm_std=None,
        dataset_dir="datasets/IMAGENET",
    ):
        """
        Parameters
        ----------
        train_csv_paths, val_csv_paths, test_csv_paths, predict_csv_paths
            各阶段 CSV 路径列表；须与 ``prepare_data`` 生成文件位置一致。详见 ``BaseDataModule``。
        transform_train, transform_val
            训练、验证集变换。详见 ``BaseDataModule``。
        transform_test, transform_predict
            测试、预测变换；默认 ``None`` 时父类回退为 ``transform_val``。
        batch_size, num_workers
            各 ``DataLoader`` 批量与 worker 数。
        key_map, predict_key_map, map_class_id_to_class_name, norm_mean, norm_std
            图像分类 DataModule 与 Dataset 共用配置；语义见 ``ImageClassifierDataModule``。
        dataset_dir : str or pathlib.Path
            ILSVRC 布局根目录（含 ``ILSVRC/`` 子树）；CSV 写在该目录下。
        """
        super().__init__(
            key_map=key_map,
            predict_key_map=predict_key_map,
            map_class_id_to_class_name=map_class_id_to_class_name,
            norm_mean=norm_mean,
            norm_std=norm_std,
            train_csv_paths=train_csv_paths,
            val_csv_paths=val_csv_paths,
            test_csv_paths=test_csv_paths,
            predict_csv_paths=predict_csv_paths,
            transform_train=transform_train,
            transform_val=transform_val,
            transform_test=transform_test,
            transform_predict=transform_predict,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.dataset_dir = Path(dataset_dir)

    @staticmethod
    def _first_synset_from_val_xml(xml_path: Path) -> str | None:
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            return None
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                return name_el.text.strip()
        return None

    def _class_synset_to_idx(self, train_dir: Path) -> dict[str, int]:
        m = self._map_spec_effective_for_csv_generation(
            self._map_class_id_to_class_name_spec
        )
        if m:
            return {name: idx for idx, name in m.items()}
        synsets = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        return {s: i for i, s in enumerate(synsets)}

    def _generate_csv_files(self) -> None:
        root = self.dataset_dir.resolve()
        train_dir = root / self._REL_DATA / "train"
        val_img_dir = root / self._REL_DATA / "val"
        val_ann_dir = root / self._REL_ANN / "val"
        sets_dir = root / self._REL_SETS

        train_csv_path = root / "train.csv"
        val_csv_path = root / "val.csv"
        test_csv_path = root / "test.csv"
        predict_csv_path = root / "predict.csv"
        map_csv_path = root / "map_class_id_to_class_name.csv"

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and test_csv_path.exists()
            and predict_csv_path.exists()
        ):
            if train_dir.is_dir() and not map_csv_path.exists():
                synset_to_idx = self._class_synset_to_idx(train_dir)
                self._write_map_class_id_to_class_name_csv(map_csv_path, synset_to_idx)
                print(f"ImageNetDataModule: wrote missing {map_csv_path}")
            print("============================================================================")
            print("ImageNetDataModule: CSV files already exist, skipping generation:")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {test_csv_path}")
            print(f"  - {predict_csv_path}")
            print(f"  - {map_csv_path}")
            print("============================================================================")
            return

        if not train_dir.is_dir():
            raise FileNotFoundError(f"ImageNet train directory not found: {train_dir}")
        if not val_img_dir.is_dir():
            raise FileNotFoundError(f"ImageNet val image directory not found: {val_img_dir}")
        test_img_dir = root / self._REL_DATA / "test"
        if not test_img_dir.is_dir():
            raise FileNotFoundError(f"ImageNet test image directory not found: {test_img_dir}")
        if not val_ann_dir.is_dir():
            raise FileNotFoundError(f"ImageNet val annotation directory not found: {val_ann_dir}")

        train_cls_txt = sets_dir / "train_cls.txt"
        val_txt = sets_dir / "val.txt"
        test_txt = sets_dir / "test.txt"
        if not train_cls_txt.is_file():
            raise FileNotFoundError(f"Missing ImageSets file: {train_cls_txt}")
        if not val_txt.is_file():
            raise FileNotFoundError(f"Missing ImageSets file: {val_txt}")
        if not test_txt.is_file():
            raise FileNotFoundError(f"Missing ImageSets file: {test_txt}")

        synset_to_idx = self._class_synset_to_idx(train_dir)

        train_rows = []
        with open(train_cls_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, _ = line.split(None, 1)
                synset, _, _ = rel.partition("/")
                if synset not in synset_to_idx:
                    continue
                rel_jpeg = f"{self._REL_DATA.as_posix()}/train/{rel}.JPEG"
                train_rows.append(
                    {
                        "path_img": rel_jpeg,
                        "class_name": synset,
                        "class_id": synset_to_idx[synset],
                    }
                )

        val_rows = []
        skipped_val = 0
        with open(val_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem, _ = line.split(None, 1)
                xml_path = val_ann_dir / f"{stem}.xml"
                synset = self._first_synset_from_val_xml(xml_path)
                if synset is None or synset not in synset_to_idx:
                    skipped_val += 1
                    continue
                rel_jpeg = f"{self._REL_DATA.as_posix()}/val/{stem}.JPEG"
                cid = synset_to_idx[synset]
                val_rows.append(
                    {"path_img": rel_jpeg, "class_name": synset, "class_id": cid}
                )

        if skipped_val:
            print(f"⚠️  Skipped {skipped_val} val rows (missing XML, parse error, or unknown synset).")

        test_rows = []
        predict_rows = []
        with open(test_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem, _ = line.split(None, 1)
                rel_jpeg = f"{self._REL_DATA.as_posix()}/test/{stem}.JPEG"
                test_rows.append({"path_img": rel_jpeg})
                predict_rows.append({"path_img": rel_jpeg})

        train_df = pd.DataFrame(train_rows, columns=["path_img", "class_name", "class_id"])
        val_df = pd.DataFrame(val_rows, columns=["path_img", "class_name", "class_id"])
        test_df = pd.DataFrame(test_rows, columns=["path_img"])
        predict_df = pd.DataFrame(predict_rows, columns=["path_img"])

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        predict_df.to_csv(predict_csv_path, index=False)
        self._write_map_class_id_to_class_name_csv(map_csv_path, synset_to_idx)

        print("============================================================================")
        print(f"Generated {train_csv_path} with {len(train_df)} entries")
        print(f"Generated {val_csv_path} with {len(val_df)} entries")
        print(f"Generated {test_csv_path} with {len(test_df)} entries")
        print(f"Generated {predict_csv_path} with {len(predict_df)} entries")
        print(f"Generated {map_csv_path} with {len(synset_to_idx)} classes")
        print("============================================================================")

    def prepare_data(self) -> None:
        """根据本地 ILSVRC 生成 CSV（不下载整库）。Lightning 顺序：``prepare_data`` → ``setup``；
        映射 CSV 在本方法中生成，在父类 ``setup`` 中再严格读入 ``map_class_id_to_class_name``。
        """
        self._generate_csv_files()
