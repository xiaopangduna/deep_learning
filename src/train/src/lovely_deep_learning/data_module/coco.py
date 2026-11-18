"""COCO 目标检测数据模块：COCO8（Ultralytics 小样）与 COCO 2017 全量（YOLO 标签 + 图像）。"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
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

COCO2017_LABELS_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels.zip"
)
COCO_TRAIN2017_IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL2017_IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TEST2017_IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/test2017.zip"

ZIP_COCO2017_LABELS = "coco2017labels.zip"
ZIP_TRAIN2017_IMAGES = "train2017.zip"
ZIP_VAL2017_IMAGES = "val2017.zip"
ZIP_TEST2017_IMAGES = "test2017.zip"

_COCO_LABEL_PREFIX = "coco/"
_COCO_IMAGE_ZIPS = (
    (ZIP_TRAIN2017_IMAGES, "train2017", COCO_TRAIN2017_IMAGES_ZIP_URL),
    (ZIP_VAL2017_IMAGES, "val2017", COCO_VAL2017_IMAGES_ZIP_URL),
    (ZIP_TEST2017_IMAGES, "test2017", COCO_TEST2017_IMAGES_ZIP_URL),
)
_COCO_SPLIT_LISTS = {"train2017": "train2017.txt", "val2017": "val2017.txt"}

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


def ensure_coco_files(dataset_dir) -> None:
    """保证 ``dataset_dir`` 里有标签和 train/val/test 图像。已有的完整 zip 与已解压目录会跳过。

    标签包去掉顶层 ``coco/`` 后全部解压。图像包解压到 ``images/``，得到
    ``images/train2017``、``images/val2017`` 和 ``images/test2017``。
    """
    root = _coco_dataset_root(dataset_dir)
    _ensure_labels(root)
    for filename, split, url in _COCO_IMAGE_ZIPS:
        _ensure_image_split(root, filename, split, url)


def _coco_dataset_root(dataset_dir) -> Path:
    path = Path(dataset_dir).expanduser()
    if path.is_symlink() and not path.exists():
        raise FileNotFoundError(f"COCO dataset_dir not found: {path}")
    if not path.exists():
        path.mkdir(parents=True)
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO dataset_dir not found: {root}")
    return root


def _ensure_zip(root: Path, filename: str, url: str) -> Path:
    zip_path = root / filename
    if zip_path.is_file() and zipfile.is_zipfile(zip_path):
        print(f"COCODataModule: 使用已有压缩包（不重新下载）: {zip_path}")
        return zip_path
    print(f"COCODataModule: 下载 {filename} -> {zip_path}")
    _download_zip(url, zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"COCODataModule: 压缩包无效: {zip_path}")
    return zip_path


def _download_zip(url: str, dest: Path) -> None:
    """下载到 ``dest``。已有不完整文件时带 Range 续传；服务端忽略 Range 则整包重下。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume = dest.is_file() and dest.stat().st_size > 0
    try:
        _stream_download(url, dest, resume=resume)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not dest.is_file():
            raise
        dest.unlink()
        _stream_download(url, dest, resume=False)


def _stream_download(url: str, dest: Path, *, resume: bool) -> None:
    existing = dest.stat().st_size if resume and dest.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None) or response.getcode()
        mode = "ab" if existing and status == 206 else "wb"
        with dest.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def _ensure_labels(root: Path) -> None:
    zip_path = _ensure_zip(root, ZIP_COCO2017_LABELS, COCO2017_LABELS_ZIP_URL)
    with zipfile.ZipFile(zip_path) as archive:
        expected = {
            split: _zip_label_txt_count(archive, split)
            for split in ("train2017", "val2017")
        }
        if any(count == 0 for count in expected.values()):
            raise RuntimeError(
                f"COCODataModule: {zip_path} 中没有 coco/labels/train2017 与 "
                "coco/labels/val2017"
            )
        if all(
            _count_named(root / "labels" / split, ".txt") >= expected[split]
            for split in expected
        ):
            print("COCO YOLO 标签已齐全，跳过解压。")
            return
        print(f"Extracting {zip_path} -> {root}（去掉 coco/ 前缀）")
        _extract_zip_strip_prefix(archive, root, _COCO_LABEL_PREFIX)


def _zip_label_txt_count(archive: zipfile.ZipFile, split: str) -> int:
    prefix = f"coco/labels/{split}/"
    return sum(
        1
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.startswith(prefix)
        and info.filename.endswith(".txt")
    )


def _count_named(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1 for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix)
    )


def _count_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file())


def _extract_zip_strip_prefix(archive: zipfile.ZipFile, root: Path, prefix: str) -> None:
    root = root.resolve()
    for info in archive.infolist():
        name = info.filename
        relative = name[len(prefix) :] if name.startswith(prefix) else name
        relative = relative.rstrip("/")
        if relative in ("", "."):
            continue
        dest = _safe_dest(root, relative)
        if info.is_dir() or name.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def _safe_dest(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"COCODataModule: 非法压缩包路径: {relative}")
    dest = (root / relative_path).resolve()
    if not dest.is_relative_to(root):
        raise RuntimeError(f"COCODataModule: 非法压缩包路径: {relative}")
    return dest


def _ensure_image_split(root: Path, filename: str, split: str, url: str) -> None:
    zip_path = _ensure_zip(root, filename, url)
    images_root = root / "images"
    dest = images_root / split
    with zipfile.ZipFile(zip_path) as archive:
        expected = sum(1 for info in archive.infolist() if not info.is_dir())
    got = _count_files(dest)
    if expected and got >= expected:
        print(f"COCO {split} 图像已齐全，跳过解压。")
        return
    images_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {images_root}")
    _extract_image_archive(zip_path, images_root)
    got = _count_files(dest)
    if got < expected:
        raise RuntimeError(
            f"COCODataModule: {split} 解压后有 {got} 个文件，压缩包内有 {expected} 个"
        )


def _extract_image_archive(zip_path: Path, images_root: Path) -> None:
    unzip = shutil.which("unzip")
    if unzip:
        completed = subprocess.run(
            [unzip, "-n", "-q", str(zip_path), "-d", str(images_root)],
            check=False,
        )
        if completed.returncode > 1:
            raise RuntimeError(
                f"COCODataModule: unzip 失败（退出码 {completed.returncode}）: {zip_path}"
            )
        return
    with zipfile.ZipFile(zip_path) as archive:
        images_root = images_root.resolve()
        for info in archive.infolist():
            relative = info.filename.rstrip("/")
            if not relative:
                continue
            dest = _safe_dest(images_root, relative)
            if info.is_dir() or info.filename.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)


def generate_coco_csvs(dataset_dir, *, overwrite: bool = False) -> None:
    """按 ``train2017.txt`` / ``val2017.txt`` 在 ``dataset_dir`` 下写 train/val/predict CSV 和类别映射。

    列表里的每张图都写入，包括没有 txt 的图。那些图的标签路径仍写成
    ``labels/{split}/{stem}.txt``，文件不存在时读作空标签（背景）。不读取 JPEG。
    ``dataset_dir`` 可以是软链接。路径列仍相对 ``dataset_dir``，不把链接展开进 CSV。
    四个文件都在且 ``overwrite`` 为 False 时跳过写入，但仍检查两个图像列表。
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO dataset_dir not found: {root}")
    _require_coco_image_lists(root)

    train_csv_path = root / "train.csv"
    val_csv_path = root / "val.csv"
    predict_csv_path = root / "predict.csv"
    map_csv_path = root / "map_class_id_to_class_name.csv"
    if (
        not overwrite
        and train_csv_path.is_file()
        and val_csv_path.is_file()
        and predict_csv_path.is_file()
        and map_csv_path.is_file()
    ):
        print("============================================================================")
        print("COCODataModule: CSV files already exist, skipping generation:")
        print(f"  - {train_csv_path}")
        print(f"  - {val_csv_path}")
        print(f"  - {predict_csv_path}")
        print(f"  - {map_csv_path}")
        print("============================================================================")
        return

    train_rows = _collect_coco_split_rows(root, "train2017")
    val_rows = _collect_coco_split_rows(root, "val2017")
    predict_rows = [{"path_img": row["path_img"]} for row in val_rows]
    class_to_idx = {name: index for index, name in enumerate(COCO80_CLASS_NAMES)}

    pd.DataFrame(train_rows, columns=["path_img", "path_label_detect_yolo"]).to_csv(
        train_csv_path, index=False
    )
    pd.DataFrame(val_rows, columns=["path_img", "path_label_detect_yolo"]).to_csv(
        val_csv_path, index=False
    )
    pd.DataFrame(predict_rows, columns=["path_img"]).to_csv(predict_csv_path, index=False)
    ObjectDetectDataModule._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

    print("============================================================================")
    print(f"Generated {train_csv_path} with {len(train_rows)} entries")
    print(f"Generated {val_csv_path} with {len(val_rows)} entries")
    print(f"Generated {predict_csv_path} with {len(predict_rows)} entries")
    print(f"Generated {map_csv_path} with {len(class_to_idx)} classes")
    print("============================================================================")


def _require_coco_image_lists(root: Path) -> None:
    for name in _COCO_SPLIT_LISTS.values():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"COCO image list not found: {path}")


def _collect_coco_split_rows(root: Path, split: str) -> list[dict]:
    """按官方图像列表写行，不打开图像。没有 txt 的图也保留，标签路径按同名 txt 书写。"""
    rows: list[dict] = []
    list_path = root / _COCO_SPLIT_LISTS[split]
    for line in list_path.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.startswith("./"):
            rel = rel[2:]
        img = Path(rel)
        if not img.parts or img.parts[0] != "images":
            raise ValueError(f"COCO image list entry must start with images/: {line}")
        label = Path("labels", *img.parts[1:]).with_suffix(".txt")
        rows.append(
            {
                "path_img": img.as_posix(),
                "path_label_detect_yolo": label.as_posix(),
            }
        )
    return rows


class COCODataModule(ObjectDetectDataModule):
    """COCO 2017 检测（80 类）：``prepare_data`` 下载并解压标签与 train/val 图像，再生成 CSV，
    并沿用目标检测 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. Lightning 在 ``fit`` / ``validate`` / ``test`` / ``predict`` 之前调用 ``prepare_data``：
       缺少 ``coco2017labels.zip``、``train2017.zip``、``val2017.zip``、``test2017.zip`` 时下载到 ``dataset_dir``。
       标签包去掉顶层 ``coco/`` 后全部解压；图像包解压到 ``images/``。已完整的 zip 和已解压的划分会跳过。
       然后按 ``train2017.txt`` / ``val2017.txt`` 生成 CSV，没有 txt 的图也写入。四个 CSV 文件已存在则跳过生成。也可先按 README 手动解压，
       再运行 ``datasets/COCO/coco_to_csv.py``，它只调用 ``generate_coco_csvs``。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ObjectDetectDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/COCO/coco``。压缩包、解压结果和 CSV 都在这一层；
      ``datasets/COCO/`` 本身只放 README 和生成脚本。``coco`` 可以是软链接。
      解压后为 ``images/train2017/*.jpg``、``images/val2017/*.jpg``、
      ``labels/train2017/*.txt``、``labels/val2017/*.txt``。
    - 生成的 CSV 写在 ``dataset_dir`` 下；路径列为相对 ``dataset_dir`` 的路径，与 ``BaseDataset`` 按 CSV 目录解析一致。
    - ``train.csv`` / ``val.csv`` 含 ``path_img`` 与 ``path_label_detect_yolo``。
      ``predict.csv`` 仅 ``path_img``，来自验证集标签。``map_class_id_to_class_name.csv`` 为 COCO 80 类。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ObjectDetectDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。
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
            含压缩包、``images/``、``labels/`` 和 CSV 的目录，默认 ``datasets/COCO/coco``。
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

    def prepare_data(self) -> None:
        """下载并解压缺失的 COCO 2017 文件，再生成 CSV。Lightning 顺序：``prepare_data`` → ``setup``。
        CSV 与 ``datasets/COCO/coco_to_csv.py`` 共用 ``generate_coco_csvs``。
        """
        ensure_coco_files(self.dataset_dir)
        generate_coco_csvs(self.dataset_dir)
