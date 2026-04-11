"""COCO 目标检测数据模块：COCO8（Ultralytics 小样）与 COCO 2017 全量（YOLO 标签 + 图像）。"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

from .object_detect import ObjectDetectDataModule

# COCO 检测 80 类名称（id 0–79），与 Ultralytics / YOLO COCO 配置一致
COCO80_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO8_DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
)

# 与 ultralytics/cfg/datasets/coco.yaml 中 download 脚本一致
COCO2017_LABELS_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels.zip"
)
COCO_TRAIN2017_IMAGES_ZIP_URL = (
    "http://images.cocodataset.org/zips/train2017.zip"
)
COCO_VAL2017_IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TEST2017_IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/test2017.zip"

ZIP_COCO2017_LABELS = "coco2017labels.zip"
ZIP_TRAIN2017_IMAGES = "train2017.zip"
ZIP_VAL2017_IMAGES = "val2017.zip"
ZIP_TEST2017_IMAGES = "test2017.zip"


def _has_txt_labels(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.txt"))


def _has_rgb_images(d: Path) -> bool:
    if not d.is_dir():
        return False
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"):
        if any(d.glob(f"*{ext}")):
            return True
    return False


def _ensure_zip(parent_dir: Path, filename: str, url: str) -> Path:
    """
    保证 ``parent_dir / filename`` 存在：若已有则直接使用（支持提前放入目录），否则从 ``url`` 下载。

    解压后 **不删除** 压缩包，便于离线复用与校验。
    """
    parent_dir.mkdir(parents=True, exist_ok=True)
    zip_path = parent_dir / filename
    if zip_path.is_file():
        print(f"COCODataModule: 使用已有压缩包（不重新下载）: {zip_path}")
        return zip_path
    print(f"COCODataModule: 下载 {filename} -> {zip_path}")
    urllib.request.urlretrieve(url, str(zip_path))
    return zip_path


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _find_yolo_label_split_dir(extract_root: Path, split: str) -> Path | None:
    """
    官方 ``coco2017labels.zip`` 内为 ``coco/labels/{train2017,val2017}/*.txt``；
    其它版本也可能为顶层 ``labels/...`` 等，此处按优先级探测。
    """
    candidates = [
        extract_root / "coco" / "labels" / split,
        extract_root / "labels" / split,
        extract_root / split,
        extract_root / "coco2017labels" / "labels" / split,
        extract_root / "coco2017labels" / split,
    ]
    for src in candidates:
        if _has_txt_labels(src):
            return src
    for src in extract_root.glob(f"**/labels/{split}"):
        if src.is_dir() and _has_txt_labels(src):
            return src
    for src in extract_root.glob(f"**/{split}"):
        if (
            src.is_dir()
            and src.name == split
            and _has_txt_labels(src)
            and "labels" in src.parts
        ):
            return src
    return None


def _normalize_coco2017_labels_layout(extract_root: Path, dataset_dir: Path) -> None:
    labels_root = dataset_dir / "labels"
    for split in ("train2017", "val2017"):
        dest = labels_root / split
        if _has_txt_labels(dest):
            continue
        src = _find_yolo_label_split_dir(extract_root, split)
        if src is None:
            raise RuntimeError(
                f"COCODataModule: 未在 {extract_root} 中找到 {split} 的 YOLO txt 标签，"
                "请检查 coco2017labels.zip 是否与 Ultralytics 官方一致。"
            )
        labels_root.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))


def _ensure_val_images(dataset_dir: Path, parent_dir: Path) -> None:
    img_val = dataset_dir / "images" / "val2017"
    if _has_rgb_images(img_val):
        print("COCO val2017 图像目录已存在，跳过解压。")
        return
    zip_path = _ensure_zip(parent_dir, ZIP_VAL2017_IMAGES, COCO_VAL2017_IMAGES_ZIP_URL)
    images_root = dataset_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    _extract_zip(zip_path, images_root)
    print("val2017 图像解压完成（压缩包已保留）。")


def _ensure_train_images(dataset_dir: Path, parent_dir: Path) -> None:
    img_train = dataset_dir / "images" / "train2017"
    if _has_rgb_images(img_train):
        print("COCO train2017 图像目录已存在，跳过解压。")
        return
    zip_path = _ensure_zip(
        parent_dir, ZIP_TRAIN2017_IMAGES, COCO_TRAIN2017_IMAGES_ZIP_URL
    )
    images_root = dataset_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    _extract_zip(zip_path, images_root)
    print("train2017 图像解压完成（体量较大；压缩包已保留）。")


def _ensure_test_images(
    dataset_dir: Path, parent_dir: Path, *, auto_download: bool
) -> None:
    img_test = dataset_dir / "images" / "test2017"
    if _has_rgb_images(img_test):
        print("COCO test2017 图像目录已存在，跳过解压。")
        return
    zip_path = parent_dir / ZIP_TEST2017_IMAGES
    if not zip_path.is_file():
        if auto_download:
            zip_path = _ensure_zip(
                parent_dir, ZIP_TEST2017_IMAGES, COCO_TEST2017_IMAGES_ZIP_URL
            )
        else:
            print(
                "COCODataModule: 未放置 test2017.zip 且未开启 test2017 自动下载，跳过 test2017。"
            )
            return
    else:
        print(f"COCODataModule: 使用已有压缩包: {zip_path}")
    images_root = dataset_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    _extract_zip(zip_path, images_root)
    print("test2017 图像解压完成（压缩包已保留）。")


def _ensure_yolo_labels(dataset_dir: Path, parent_dir: Path) -> None:
    lbl_train = dataset_dir / "labels" / "train2017"
    lbl_val = dataset_dir / "labels" / "val2017"
    if _has_txt_labels(lbl_train) and _has_txt_labels(lbl_val):
        print("COCO YOLO 标签目录已存在，跳过解压。")
        return
    zip_path = _ensure_zip(
        parent_dir, ZIP_COCO2017_LABELS, COCO2017_LABELS_ZIP_URL
    )
    tmp = parent_dir / "_tmp_coco2017labels"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _extract_zip(zip_path, tmp)
        _normalize_coco2017_labels_layout(tmp, dataset_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("coco2017labels 解压完成（压缩包已保留）。")


class COCO8DataModule(ObjectDetectDataModule):
    """Ultralytics COCO8：下载并解压数据、生成 CSV，并沿用目标检测 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. ``trainer.fit`` 之前 Lightning 会调用 ``prepare_data``：若 ``dataset_dir/images/train`` 等尚不存在则
       从 ``download_url``（默认 ``COCO8_DOWNLOAD_URL``）下载 zip 并解压到 ``dataset_dir`` 的父目录，
       然后在父目录下写入 ``train.csv`` / ``val.csv`` / ``predict.csv`` 及 ``map_class_id_to_class_name.csv``
       （若三个主 CSV 已存在则跳过整批生成；仅缺映射文件时会补写映射）。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ObjectDetectDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/COCO8/coco8``，即解压后含 ``images/train``、``images/val``、
      ``labels/train``、``labels/val`` 的根目录（与官方 zip 内 ``coco8/`` 布局一致）。
    - 生成的 CSV 中 ``path_img``、``path_label_detect_yolo`` 为相对路径（如 ``coco8/images/train/...``），
      相对 **CSV 文件所在目录** 解析；因此 YAML 里 ``train_csv_paths`` 等应列为 ``datasets/COCO8/train.csv``
      这类与解压布局一致的路径。
    - ``predict.csv`` 仅含 ``path_img``，默认取验证集图像（与 ``ImageNetteDataModule`` 一致）。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ObjectDetectDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。``map_class_id_to_class_name`` 在 ``setup`` 中的解析语义见父类；
    ``prepare_data`` 写出的类别映射表为 COCO 80 类标准名称与 id。
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
        batch_size: int = 32,
        num_workers: int = 4,
        key_map=None,
        predict_key_map=None,
        map_class_id_to_class_name=None,
        norm_mean=None,
        norm_std=None,
        dataset_dir: str = "datasets/COCO8/coco8",
        download_url: str | None = None,
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
            目标检测 DataModule 与 Dataset 共用配置；语义见 ``ObjectDetectDataModule``。
        dataset_dir : str or pathlib.Path
            COCO8 解压后的根目录（含 ``images/``、``labels/``）。zip 下载至其父目录并解压到该父目录下。
        download_url
            数据集 zip 地址；默认 ``COCO8_DOWNLOAD_URL``（Ultralytics 官方 coco8.zip）。
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
        self.download_url = download_url or COCO8_DOWNLOAD_URL

    def prepare_data(self):
        """若本地尚无 COCO8 目录则下载并解压 zip，再生成或跳过已存在的 CSV 与映射文件。"""
        parent_dir = self.dataset_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        train_img_dir = self.dataset_dir / "images" / "train"
        val_img_dir = self.dataset_dir / "images" / "val"

        if not (train_img_dir.is_dir() and val_img_dir.is_dir()):
            zip_path = parent_dir / "coco8.zip"
            print(f"Downloading COCO8 dataset to {zip_path} ...")
            urllib.request.urlretrieve(self.download_url, str(zip_path))

            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(path=str(parent_dir))
            zip_path.unlink(missing_ok=True)
            print("Download and extraction complete.")
        else:
            print("COCO8 dataset already exists, skipping download.")

        self._generate_csv_files(self.dataset_dir)

    @staticmethod
    def _coco_class_name_to_idx() -> dict[str, int]:
        return {name: i for i, name in enumerate(COCO80_CLASS_NAMES)}

    def _collect_split_rows(self, dataset_dir: Path, split: str) -> list[dict]:
        """``split`` 为 ``train`` 或 ``val``：生成带 ``path_img``、``path_label_detect_yolo`` 的行。"""
        prefix = dataset_dir.name
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        rows: list[dict] = []
        if not img_dir.is_dir():
            return rows
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.is_file():
                continue
            rows.append(
                {
                    "path_img": f"{prefix}/images/{split}/{img_path.name}",
                    "path_label_detect_yolo": f"{prefix}/labels/{split}/{stem}.txt",
                }
            )
        return rows

    def _generate_csv_files(self, dataset_dir: Path):
        """生成 ``train.csv``、``val.csv``、``predict.csv`` 及 ``map_class_id_to_class_name.csv``。"""
        dataset_dir = Path(dataset_dir)
        output_dir = dataset_dir.parent
        output_dir.mkdir(exist_ok=True)

        train_csv_path = output_dir / "train.csv"
        val_csv_path = output_dir / "val.csv"
        predict_csv_path = output_dir / "predict.csv"
        map_csv_path = output_dir / "map_class_id_to_class_name.csv"
        train_split_dir = dataset_dir / "images" / "train"

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and predict_csv_path.exists()
        ):
            if train_split_dir.is_dir() and not map_csv_path.exists():
                class_to_idx = self._coco_class_name_to_idx()
                self._write_map_class_id_to_class_name_csv(
                    map_csv_path, class_to_idx
                )
                print(f"COCO8DataModule: wrote missing {map_csv_path}")
            print(
                "============================================================================"
            )
            print("COCO8DataModule: CSV files already exist, skipping generation")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {predict_csv_path}")
            print(f"  - {map_csv_path}")
            print(
                "============================================================================"
            )
            return

        class_to_idx = self._coco_class_name_to_idx()
        train_rows = self._collect_split_rows(dataset_dir, "train")
        val_rows = self._collect_split_rows(dataset_dir, "val")

        predict_rows = [{"path_img": r["path_img"]} for r in val_rows]

        pd.DataFrame(
            train_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(train_csv_path, index=False)
        pd.DataFrame(
            val_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(val_csv_path, index=False)
        pd.DataFrame(predict_rows, columns=["path_img"]).to_csv(
            predict_csv_path, index=False
        )
        self._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

        print(
            "============================================================================"
        )
        print(f"Generated {train_csv_path} with {len(train_rows)} entries")
        print(f"Generated {val_csv_path} with {len(val_rows)} entries")
        print(f"Generated {predict_csv_path} with {len(predict_rows)} entries")
        print(f"Generated {map_csv_path} with {len(class_to_idx)} classes")
        print(
            "============================================================================"
        )


class COCODataModule(ObjectDetectDataModule):
    """
    COCO 2017 检测（80 类）：``prepare_data`` 解压标签与图像并生成 CSV。

    **压缩包位置（``dataset_dir`` 父目录，默认 ``datasets/COCO/``）**

    可将官方文件 **提前放入** 该目录，文件名与下表一致，则 **不会重新下载**：

    - ``coco2017labels.zip`` — Ultralytics YOLO 标签
    - ``train2017.zip`` / ``val2017.zip`` — COCO 图像
    - ``test2017.zip``（可选）— 仅 test 图像；放入则解压；未放入且未设 ``download_test2017`` 则跳过

    若某 zip 不存在，则从与 Ultralytics ``coco.yaml`` 一致的 URL **自动下载**（``test2017`` 默认不自动下，
    见 ``download_test2017``）。解压后 **保留** 所有压缩包，便于断点续传与离线复用。

    **解压后目录**（``dataset_dir`` 例如 ``datasets/COCO/coco``）::

        {dataset_dir}/images/train2017/*.jpg
        {dataset_dir}/images/val2017/*.jpg
        {dataset_dir}/images/test2017/*.jpg   # 仅当提供 test2017.zip 并已解压
        {dataset_dir}/labels/train2017/*.txt
        {dataset_dir}/labels/val2017/*.txt

    CSV 与映射写在 ``dataset_dir`` 父目录：``train.csv``、``val.csv``、``predict.csv``、
    ``map_class_id_to_class_name.csv``。

    体量参考：标签 zip 较小；``val2017`` 约 1GB；``train2017`` 约 19GB；``test2017`` 约 6GB。
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
        batch_size: int = 16,
        num_workers: int = 4,
        key_map=None,
        predict_key_map=None,
        map_class_id_to_class_name=None,
        norm_mean=None,
        norm_std=None,
        dataset_dir: str = "datasets/COCO/coco",
        download_test2017: bool = False,
    ):
        """
        Parameters
        ----------
        download_test2017
            若为 True，当 ``{dataset_dir 父目录}/test2017.zip`` 不存在时从官网下载（约 6GB）。
            默认 False：仅当该 zip 已手动放入父目录时才解压 test2017。
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
        self.download_test2017 = bool(download_test2017)

    @staticmethod
    def _coco_class_name_to_idx() -> dict[str, int]:
        return {name: i for i, name in enumerate(COCO80_CLASS_NAMES)}

    def prepare_data(self) -> None:
        parent_dir = self.dataset_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        _ensure_yolo_labels(self.dataset_dir, parent_dir)
        _ensure_val_images(self.dataset_dir, parent_dir)
        _ensure_train_images(self.dataset_dir, parent_dir)
        _ensure_test_images(
            self.dataset_dir, parent_dir, auto_download=self.download_test2017
        )

        self._generate_csv_files(self.dataset_dir)

    def _collect_split_rows(self, dataset_dir: Path, split: str) -> list[dict]:
        """``split`` 为 ``train2017`` 或 ``val2017``。"""
        prefix = dataset_dir.name
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        rows: list[dict] = []
        if not img_dir.is_dir():
            return rows
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.is_file():
                continue
            rows.append(
                {
                    "path_img": f"{prefix}/images/{split}/{img_path.name}",
                    "path_label_detect_yolo": (
                        f"{prefix}/labels/{split}/{stem}.txt"
                    ),
                }
            )
        return rows

    def _generate_csv_files(self, dataset_dir: Path) -> None:
        dataset_dir = Path(dataset_dir)
        output_dir = dataset_dir.parent
        output_dir.mkdir(exist_ok=True)

        train_csv_path = output_dir / "train.csv"
        val_csv_path = output_dir / "val.csv"
        predict_csv_path = output_dir / "predict.csv"
        map_csv_path = output_dir / "map_class_id_to_class_name.csv"
        train_split_dir = dataset_dir / "images" / "train2017"

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and predict_csv_path.exists()
        ):
            if train_split_dir.is_dir() and not map_csv_path.exists():
                class_to_idx = self._coco_class_name_to_idx()
                self._write_map_class_id_to_class_name_csv(
                    map_csv_path, class_to_idx
                )
                print(f"COCODataModule: wrote missing {map_csv_path}")
            print("=" * 76)
            print("COCODataModule: CSV files already exist, skipping generation")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {predict_csv_path}")
            print(f"  - {map_csv_path}")
            print("=" * 76)
            return

        class_to_idx = self._coco_class_name_to_idx()
        train_rows = self._collect_split_rows(dataset_dir, "train2017")
        val_rows = self._collect_split_rows(dataset_dir, "val2017")
        predict_rows = [{"path_img": r["path_img"]} for r in val_rows]

        pd.DataFrame(
            train_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(train_csv_path, index=False)
        pd.DataFrame(
            val_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(val_csv_path, index=False)
        pd.DataFrame(predict_rows, columns=["path_img"]).to_csv(
            predict_csv_path, index=False
        )
        self._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

        print("=" * 76)
        print(f"Generated {train_csv_path} with {len(train_rows)} entries")
        print(f"Generated {val_csv_path} with {len(val_rows)} entries")
        print(f"Generated {predict_csv_path} with {len(predict_rows)} entries")
        print(f"Generated {map_csv_path} with {len(class_to_idx)} classes")
        print("=" * 76)
