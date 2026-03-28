import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .image_classifier import ImageClassifierDataModule


class ImageNetDataModule(ImageClassifierDataModule):
    """
    ImageNet ILSVRC CLS-LOC：不下载数据，在 prepare_data 中生成 train/val/test/predict 的 CSV。

    期望目录（dataset_dir 为 IMAGENET 根目录，例如 datasets/IMAGENET）：
      ILSVRC/Data/CLS-LOC/train/<synset>/*.JPEG
      ILSVRC/Data/CLS-LOC/val/*.JPEG
      ILSVRC/Data/CLS-LOC/test/*.JPEG
      ILSVRC/Annotations/CLS-LOC/val/*.xml
      ILSVRC/ImageSets/CLS-LOC/train_cls.txt
      ILSVRC/ImageSets/CLS-LOC/val.txt
      ILSVRC/ImageSets/CLS-LOC/test.txt

    train.csv / val.csv：与原先相同（train_cls + val XML 标签）。
    test.csv / predict.csv：仅 path_img，来自 CLS-LOC/test（test.txt 列表）。

    CSV 写在 dataset_dir 下；path_img 为相对 dataset_dir 的路径，与 BaseDataset 按 CSV 目录解析一致。
    """

    _REL_DATA = Path("ILSVRC/Data/CLS-LOC")
    _REL_ANN = Path("ILSVRC/Annotations/CLS-LOC")
    _REL_SETS = Path("ILSVRC/ImageSets/CLS-LOC")

    def __init__(self, dataset_dir="datasets/IMAGENET", map_class_id_to_class_name=None, **kwargs):
        super().__init__(map_class_id_to_class_name=map_class_id_to_class_name, **kwargs)
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
        if self.map_class_id_to_class_name:
            return {name: idx for idx, name in self.map_class_id_to_class_name.items()}
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

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and test_csv_path.exists()
            and predict_csv_path.exists()
        ):
            print("============================================================================")
            print("CSV files already exist, skipping generation:")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {test_csv_path}")
            print(f"  - {predict_csv_path}")
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

        print("============================================================================")
        print(f"Generated {train_csv_path} with {len(train_df)} entries")
        print(f"Generated {val_csv_path} with {len(val_df)} entries")
        print(f"Generated {test_csv_path} with {len(test_df)} entries")
        print(f"Generated {predict_csv_path} with {len(predict_df)} entries")
        print("============================================================================")

    def prepare_data(self) -> None:
        self._generate_csv_files()
