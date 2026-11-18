import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import pandas as pd

from .image_classifier import ImageClassifierDataModule

_REL_DATA = Path("ILSVRC/Data/CLS-LOC")
_REL_ANN = Path("ILSVRC/Annotations/CLS-LOC")
_REL_SETS = Path("ILSVRC/ImageSets/CLS-LOC")


def generate_imagenet_csvs(dataset_dir, *, overwrite: bool = False) -> None:
    """按 ILSVRC ImageSets 与 val XML 在 ``dataset_dir`` 下写 train/val/test/predict CSV。

    不读取 JPEG。``dataset_dir/ILSVRC`` 可以是指向官方树的软链接。``path_img`` 仍相对
    ``dataset_dir``，不把链接展开进 CSV。四个 CSV 都在且 ``overwrite`` 为 False 时跳过写入，
    但仍检查 ImageSets 与 val 标注。
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet dataset_dir not found: {root}")
    _check_ilsvrc_link(root)
    _require_imagenet_metadata(root)

    train_csv_path = root / "train.csv"
    val_csv_path = root / "val.csv"
    test_csv_path = root / "test.csv"
    predict_csv_path = root / "predict.csv"
    if (
        not overwrite
        and train_csv_path.exists()
        and val_csv_path.exists()
        and test_csv_path.exists()
        and predict_csv_path.exists()
    ):
        print("============================================================================")
        print("ImageNetDataModule: CSV files already exist, skipping generation:")
        print(f"  - {train_csv_path}")
        print(f"  - {val_csv_path}")
        print(f"  - {test_csv_path}")
        print(f"  - {predict_csv_path}")
        print("============================================================================")
        return

    val_ann_dir = root / _REL_ANN / "val"
    sets_dir = root / _REL_SETS
    synset_to_idx = _class_synset_to_idx(sets_dir / "train_cls.txt")

    train_rows = []
    with open(sets_dir / "train_cls.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, _ = line.split(None, 1)
            synset, _, _ = rel.partition("/")
            if synset not in synset_to_idx:
                continue
            rel_jpeg = f"{_REL_DATA.as_posix()}/train/{rel}.JPEG"
            train_rows.append(
                {
                    "path_img": rel_jpeg,
                    "class_name": synset,
                    "class_id": synset_to_idx[synset],
                }
            )

    val_rows = []
    skipped_val = 0
    with open(sets_dir / "val.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem, _ = line.split(None, 1)
            synset = _first_synset_from_val_xml(val_ann_dir / f"{stem}.xml")
            if synset is None or synset not in synset_to_idx:
                skipped_val += 1
                continue
            rel_jpeg = f"{_REL_DATA.as_posix()}/val/{stem}.JPEG"
            val_rows.append(
                {
                    "path_img": rel_jpeg,
                    "class_name": synset,
                    "class_id": synset_to_idx[synset],
                }
            )

    if skipped_val:
        print(f"⚠️  Skipped {skipped_val} val rows (missing XML, parse error, or unknown synset).")

    test_rows = []
    with open(sets_dir / "test.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem, _ = line.split(None, 1)
            test_rows.append({"path_img": f"{_REL_DATA.as_posix()}/test/{stem}.JPEG"})

    pd.DataFrame(train_rows, columns=["path_img", "class_name", "class_id"]).to_csv(
        train_csv_path, index=False
    )
    pd.DataFrame(val_rows, columns=["path_img", "class_name", "class_id"]).to_csv(
        val_csv_path, index=False
    )
    pd.DataFrame(test_rows, columns=["path_img"]).to_csv(test_csv_path, index=False)
    pd.DataFrame(test_rows, columns=["path_img"]).to_csv(predict_csv_path, index=False)

    print("============================================================================")
    print(f"Generated {train_csv_path} with {len(train_rows)} entries")
    print(f"Generated {val_csv_path} with {len(val_rows)} entries")
    print(f"Generated {test_csv_path} with {len(test_rows)} entries")
    print(f"Generated {predict_csv_path} with {len(test_rows)} entries")
    print("============================================================================")


def _check_ilsvrc_link(root: Path) -> None:
    ilsvrc = root / "ILSVRC"
    if not ilsvrc.is_symlink():
        return
    link = os.readlink(ilsvrc)
    if not ilsvrc.exists():
        raise FileNotFoundError(f"ImageNet ILSVRC symlink is broken: {ilsvrc} -> {link}")
    print(f"ImageNet: ILSVRC -> {ilsvrc.resolve()}")


def _require_imagenet_metadata(root: Path) -> None:
    ann_dir = root / _REL_ANN / "val"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val annotation directory not found: {ann_dir}")
    for name in ("train_cls.txt", "val.txt", "test.txt"):
        path = root / _REL_SETS / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing ImageSets file: {path}")


def _class_synset_to_idx(train_cls_txt: Path) -> dict[str, int]:
    """``class_id`` 取 ``train_cls.txt`` 中 synset 的字典序，不列出 train 图片目录。"""
    synsets: set[str] = set()
    with open(train_cls_txt, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel = line.split(None, 1)[0]
            synset, _, _ = rel.partition("/")
            if synset:
                synsets.add(synset)
    return {name: index for index, name in enumerate(sorted(synsets))}


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


class ImageNetDataModule(ImageClassifierDataModule):
    """ImageNet ILSVRC CLS-LOC：不下载数据，在 ``prepare_data`` 中生成 train/val/test/predict 的 CSV，
    并沿用图像分类 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. ``trainer.fit`` 之前 Lightning 会调用 ``prepare_data``：在 ``dataset_dir`` 下生成 train/val/test/predict 四个 CSV。
       synset id 按 ``train_cls.txt`` 中的 synset 字典序分配，不读取 JPEG。主 CSV 已存在则跳过生成。也可先运行
       ``datasets/IMAGENET/imagenet_to_csv.py``，它调用同一个 ``generate_imagenet_csvs``。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ImageClassifierDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/IMAGENET/imagenet``。压缩包、解压结果和 CSV 都在这一层；
      ``datasets/IMAGENET/`` 本身只放 README 和生成脚本。``imagenet`` 可以是软链接。
      生成 CSV 需要 ``ILSVRC/Annotations/CLS-LOC/val/*.xml`` 和
      ``ILSVRC/ImageSets/CLS-LOC/train_cls.txt``、``val.txt``、``test.txt``。
      图片目录可尚未解压，``path_img`` 仍按 ImageSets 写成 ``ILSVRC/Data/CLS-LOC/...``。
    - 生成的 CSV 写在 ``dataset_dir`` 下；``path_img`` 为相对 ``dataset_dir`` 的路径，与 ``BaseDataset`` 按 CSV 目录解析一致。
    - ``train.csv`` / ``val.csv``：来自 ``train_cls`` 与 val XML 标签；``test.csv`` / ``predict.csv``：仅 ``path_img``，来自 ``test.txt`` 列表。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ImageClassifierDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。synset id 按 ``train_cls.txt`` 中的 synset 字典序分配。
    """

    _REL_DATA = _REL_DATA
    _REL_ANN = _REL_ANN
    _REL_SETS = _REL_SETS

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
        norm_mean=None,
        norm_std=None,
        dataset_dir="datasets/IMAGENET/imagenet",
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
        key_map, predict_key_map, norm_mean, norm_std
            图像分类 DataModule 与 Dataset 共用配置；语义见 ``ImageClassifierDataModule``。
        dataset_dir : str or pathlib.Path
            含压缩包、``ILSVRC/`` 和 CSV 的目录，默认 ``datasets/IMAGENET/imagenet``。
        """
        super().__init__(
            key_map=key_map,
            predict_key_map=predict_key_map,
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

    def prepare_data(self) -> None:
        """根据本地 ILSVRC 生成 CSV（不下载整库）。Lightning 顺序：``prepare_data`` → ``setup``。
        类别名在 ``setup`` 里从带标签的 CSV 推导。与 ``datasets/IMAGENET/imagenet_to_csv.py`` 共用
        ``generate_imagenet_csvs``。
        """
        generate_imagenet_csvs(self.dataset_dir)
