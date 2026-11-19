"""COCO 目标检测数据模块：COCO8（Ultralytics 小样）与 COCO 2017 全量（YOLO 标签 + 图像）。"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List

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
_COCO_TEST_LIST = "test-dev2017.txt"
_COCO_READY_DIRS = (
    Path("labels/train2017"),
    Path("labels/val2017"),
    Path("images/train2017"),
    Path("images/val2017"),
    Path("images/test2017"),
)
_COCO_BAR = "=" * 70

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


def coco_splits_ready(dataset_dir) -> bool:
    """五个划分目录里各有至少一个文件时，数据视为已准备好。"""
    root = Path(dataset_dir).expanduser()
    if root.is_symlink() and not root.exists():
        return False
    return all(_dir_has_file(root / relative) for relative in _COCO_READY_DIRS)


def ensure_coco_files(dataset_dir) -> None:
    """下载缺失的 zip 并解压还没有文件的划分。标签包去掉顶层 ``coco/``。

    某个划分目录里已经有文件就不再解压该划分。调用方在五个划分都有文件时不应进入这里。
    终端先打 ``dataset_dir``，再打压缩包和解压两块；CSV 由 ``generate_coco_csvs`` 接着打。
    """
    print(_COCO_BAR)
    print(f"COCO  {_dataset_dir_label(dataset_dir)}")
    print("压缩包")
    root = _coco_dataset_root(dataset_dir)
    jobs = [(ZIP_COCO2017_LABELS, COCO2017_LABELS_ZIP_URL)]
    jobs.extend((filename, url) for filename, _split, url in _COCO_IMAGE_ZIPS)
    name_width = max(len(name) for name, _url in jobs)
    zip_paths: dict[str, Path] = {}
    for name, url in jobs:
        zip_path, status = _zip_action(root, name)
        if status == "已有":
            size = _format_size(zip_path.stat().st_size)
            print(f"{name:<{name_width}}  {status}  {size:>8}")
        else:
            prefix = f"{name:<{name_width}}  {status}"
            _download_zip(url, zip_path, prefix=prefix)
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError(f"COCODataModule: 压缩包无效: {zip_path}")
        zip_paths[name] = zip_path

    extract_rows = [("labels", "解压，去掉 coco/ 前缀", True)]
    for _filename, split, _url in _COCO_IMAGE_ZIPS:
        needed = not _dir_has_file(root / "images" / split)
        extract_rows.append((f"images/{split}", "解压" if needed else "跳过", needed))
    if _labels_ready(root):
        extract_rows[0] = ("labels", "跳过", False)
    name_width = max(len(name) for name, _status, _needed in extract_rows)
    print(_COCO_BAR)
    print("解压")
    label_name, label_status, extract_labels = extract_rows[0]
    if extract_labels:
        count, _size = _archive_file_stats(zip_paths[ZIP_COCO2017_LABELS])
        _run_timed_row(
            f"{label_name:<{name_width}}  {label_status}   {count} 个文件",
            lambda: _extract_labels(root, zip_paths[ZIP_COCO2017_LABELS]),
        )
    else:
        print(f"{label_name:<{name_width}}  {label_status}")
    for name, status, needed in extract_rows[1:]:
        if not needed:
            print(f"{name:<{name_width}}  {status}")
            continue
        split = name.split("/", 1)[1]
        count, size = _archive_file_stats(zip_paths[f"{split}.zip"])
        (root / "images").mkdir(parents=True, exist_ok=True)

        def extract_split(zip_name: str = f"{split}.zip", split_name: str = name) -> None:
            _extract_image_archive(zip_paths[zip_name], root / "images")
            if not _dir_has_file(root / "images" / split_name.split("/", 1)[1]):
                raise RuntimeError(f"COCODataModule: {split_name} 解压后没有文件")

        _run_timed_row(
            f"{name:<{name_width}}  解压中   {count} 个文件   {_format_size(size)}",
            extract_split,
            done=f"{name:<{name_width}}  解压   {count} 个文件   {_format_size(size)}",
        )


def _dataset_dir_label(dataset_dir) -> str:
    path = Path(dataset_dir).expanduser()
    shown = path.as_posix()
    if path.is_symlink() and path.exists():
        return f"{shown} -> {path.resolve().as_posix()}"
    return shown


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1e9:.1f}GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f}MB"
    if n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


def _format_speed(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1e9:
        return f"{bytes_per_sec / 1e9:.1f}GB/s"
    if bytes_per_sec >= 1e6:
        return f"{bytes_per_sec / 1e6:.0f}MB/s"
    if bytes_per_sec >= 1e3:
        return f"{bytes_per_sec / 1e3:.0f}KB/s"
    return f"{bytes_per_sec:.0f}B/s"


def _format_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _percent(done: int, total: int | None) -> int | None:
    if not total:
        return None
    return min(int(done * 100 / total), 100)


def _transfer_text(
    label: str,
    done: int,
    total: int | None,
    written: int,
    started: float,
    *,
    final: bool,
) -> str:
    elapsed = time.monotonic() - started
    speed = written / elapsed if elapsed > 0 else 0.0
    speed_text = _format_speed(speed)
    if final and total:
        return (
            f"{label}  {_format_size(done):>8}   {'100%':>4}   {speed_text:>7}"
            f"   {_format_elapsed(elapsed)}"
        )
    if total:
        percent = min(int(done * 100 / total), 100)
        return (
            f"{label}  {_format_size(done):>8} / {_format_size(total):<8}  {percent:>3}%"
            f"   {speed_text}"
        )
    text = f"{label}  {_format_size(done):>8}   {speed_text}"
    if final:
        text += f"   {_format_elapsed(elapsed)}"
    return text


class _LiveLine:
    """终端上原地改一行。重定向到文件时，下载每过 10% 另起一行，结束再打定稿。"""

    def __init__(self) -> None:
        self._tty = sys.stdout.isatty()
        self._last = ""
        self._next_tenth = 10
        self._last_emit = time.monotonic()

    def show(
        self,
        text: str,
        *,
        final: bool = False,
        percent: int | None = None,
        force: bool = False,
    ) -> None:
        if self._tty:
            pad = " " * max(len(self._last) - len(text), 0)
            print(f"\r{text}{pad}", end="\n" if final else "", flush=True)
            self._last = "" if final else text
            return
        if final or force:
            print(text)
            return
        now = time.monotonic()
        emit = False
        if percent is None:
            emit = now - self._last_emit >= 10
        else:
            tenth = percent // 10 * 10
            emit = tenth >= self._next_tenth and tenth < 100
            if emit:
                self._next_tenth = tenth + 10
        if emit:
            print(text)
            self._last_emit = now


def _run_timed_row(working: str, action: Callable[[], None], *, done: str | None = None) -> None:
    line = _LiveLine()
    line.show(working, force=True)
    started = time.monotonic()
    action()
    elapsed = _format_elapsed(time.monotonic() - started)
    line.show(f"{done or working}   {elapsed}", final=True)


def _archive_file_stats(zip_path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.endswith("/"):
                continue
            count += 1
            size += info.file_size
    return count, size


def _content_total(response, already: int) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        length = int(raw)
    except (TypeError, ValueError):
        return None
    return already + length


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


def _dir_has_file(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(path.is_file() for path in directory.iterdir())


def _labels_ready(root: Path) -> bool:
    return _dir_has_file(root / "labels" / "train2017") and _dir_has_file(
        root / "labels" / "val2017"
    )


def _zip_action(root: Path, filename: str) -> tuple[Path, str]:
    zip_path = root / filename
    if zip_path.is_file() and zipfile.is_zipfile(zip_path):
        return zip_path, "已有"
    if zip_path.is_file() and zip_path.stat().st_size > 0:
        return zip_path, "续传"
    return zip_path, "下载"


def _download_zip(url: str, dest: Path, *, prefix: str | None = None) -> None:
    """下载到 ``dest``。已有不完整文件时带 Range 续传；服务端忽略 Range 则整包重下。

    ``prefix`` 是进度行左侧的文件名和状态。同一行在终端里刷新；输出不是终端时每过 10% 另起一行。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = prefix or f"{dest.name}  下载"
    line = _LiveLine()
    started = time.monotonic()
    state = {"written": 0, "total": None}

    def on_chunk(done: int, total: int | None, written: int) -> None:
        state["written"] = written
        state["total"] = total
        line.show(
            _transfer_text(label, done, total, state["written"], started, final=False),
            percent=_percent(done, total),
        )

    resume = dest.is_file() and dest.stat().st_size > 0
    try:
        _stream_download(url, dest, resume=resume, on_chunk=on_chunk)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not dest.is_file():
            raise
        dest.unlink()
        state["written"] = 0
        state["total"] = None
        _stream_download(url, dest, resume=False, on_chunk=on_chunk)
    done = dest.stat().st_size if dest.is_file() else 0
    line.show(
        _transfer_text(label, done, state["total"], state["written"], started, final=True),
        final=True,
    )


def _stream_download(
    url: str,
    dest: Path,
    *,
    resume: bool,
    on_chunk: Callable[[int, int | None, int], None] | None = None,
) -> None:
    existing = dest.stat().st_size if resume and dest.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None) or response.getcode()
        appending = bool(existing and status == 206)
        mode = "ab" if appending else "wb"
        base = existing if appending else 0
        total = _content_total(response, base)
        done = base
        written = 0
        if on_chunk is not None:
            on_chunk(done, total, written)
        with dest.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                written += len(chunk)
                if on_chunk is not None:
                    on_chunk(done, total, written)


def _extract_labels(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = {info.filename for info in archive.infolist()}
        if not any(name.startswith("coco/labels/train2017/") and name.endswith(".txt") for name in names):
            raise RuntimeError(
                f"COCODataModule: {zip_path} 中没有 coco/labels/train2017"
            )
        if not any(name.startswith("coco/labels/val2017/") and name.endswith(".txt") for name in names):
            raise RuntimeError(
                f"COCODataModule: {zip_path} 中没有 coco/labels/val2017"
            )
        _extract_zip_strip_prefix(archive, root, _COCO_LABEL_PREFIX)
    if not _labels_ready(root):
        raise RuntimeError(f"COCODataModule: 标签解压后划分目录里没有文件: {zip_path}")


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


def generate_coco_csvs(dataset_dir, *, overwrite: bool = False, with_header: bool = True) -> None:
    """按图像列表在 ``dataset_dir`` 下写 train/val/test/predict CSV 和类别映射。

    ``train2017.txt`` / ``val2017.txt`` 里的每张图都写入，包括没有 txt 的图。那些图的标签路径仍写成
    ``labels/{split}/{stem}.txt``，文件不存在时读作空标签（背景）。``test.csv`` 来自
    ``test-dev2017.txt``，只有 ``path_img``。不读取 JPEG。
    ``dataset_dir`` 可以是软链接。路径列仍相对 ``dataset_dir``，不把链接展开进 CSV。
    五个文件都在且 ``overwrite`` 为 False 时跳过写入，但仍检查三份图像列表。
    ``with_header`` 为 True 时先打数据目录；``prepare_data`` 已打过目录时传 False。
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO dataset_dir not found: {root}")
    _require_coco_image_lists(root)

    train_csv_path = root / "train.csv"
    val_csv_path = root / "val.csv"
    test_csv_path = root / "test.csv"
    predict_csv_path = root / "predict.csv"
    map_csv_path = root / "map_class_id_to_class_name.csv"
    outputs = (
        (train_csv_path, None),
        (val_csv_path, None),
        (test_csv_path, None),
        (predict_csv_path, None),
        (map_csv_path, None),
    )
    if not overwrite and all(path.is_file() for path, _count in outputs):
        _print_csv_block(
            [(path.name, None, "跳过") for path, _count in outputs],
            dataset_dir,
            with_header=with_header,
        )
        return

    train_rows = _collect_coco_split_rows(root, "train2017")
    val_rows = _collect_coco_split_rows(root, "val2017")
    test_rows = _collect_coco_image_rows(root, _COCO_TEST_LIST)
    predict_rows = [{"path_img": row["path_img"]} for row in val_rows]
    class_to_idx = {name: index for index, name in enumerate(COCO80_CLASS_NAMES)}

    pd.DataFrame(train_rows, columns=["path_img", "path_label_detect_yolo"]).to_csv(
        train_csv_path, index=False
    )
    pd.DataFrame(val_rows, columns=["path_img", "path_label_detect_yolo"]).to_csv(
        val_csv_path, index=False
    )
    pd.DataFrame(test_rows, columns=["path_img"]).to_csv(test_csv_path, index=False)
    pd.DataFrame(predict_rows, columns=["path_img"]).to_csv(predict_csv_path, index=False)
    ObjectDetectDataModule._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

    _print_csv_block(
        [
            (train_csv_path.name, len(train_rows), "写入"),
            (val_csv_path.name, len(val_rows), "写入"),
            (test_csv_path.name, len(test_rows), "写入"),
            (predict_csv_path.name, len(predict_rows), "写入"),
            (map_csv_path.name, len(class_to_idx), "写入"),
        ],
        dataset_dir,
        with_header=with_header,
    )


def _print_csv_block(
    rows: list[tuple[str, int | None, str]],
    dataset_dir,
    *,
    with_header: bool,
) -> None:
    print(_COCO_BAR)
    if with_header:
        print(f"COCO  {_dataset_dir_label(dataset_dir)}")
    print("CSV")
    name_width = max(len(name) for name, _count, _action in rows)
    counts = [count for _name, count, _action in rows if count is not None]
    count_width = max((len(str(count)) for count in counts), default=0)
    for name, count, action in rows:
        if count is None:
            print(f"{name:<{name_width}}  {action}")
        else:
            print(f"{name:<{name_width}}  {count:>{count_width}}  {action}")
    print(_COCO_BAR)


def _require_coco_image_lists(root: Path) -> None:
    names = (*_COCO_SPLIT_LISTS.values(), _COCO_TEST_LIST)
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"COCO image list not found: {path}")


def _iter_coco_list_images(root: Path, list_name: str):
    list_path = root / list_name
    for line in list_path.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.startswith("./"):
            rel = rel[2:]
        img = Path(rel)
        if not img.parts or img.parts[0] != "images":
            raise ValueError(f"COCO image list entry must start with images/: {line}")
        yield img


def _collect_coco_image_rows(root: Path, list_name: str) -> list[dict]:
    """只写 ``path_img``。``test-dev2017.txt`` 没有检测框。"""
    return [{"path_img": img.as_posix()} for img in _iter_coco_list_images(root, list_name)]


def _collect_coco_split_rows(root: Path, split: str) -> list[dict]:
    """按官方图像列表写行，不打开图像。没有 txt 的图也保留，标签路径按同名 txt 书写。"""
    rows: list[dict] = []
    for img in _iter_coco_list_images(root, _COCO_SPLIT_LISTS[split]):
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
       ``labels/train2017``、``labels/val2017``、``images/train2017``、``images/val2017``、
       ``images/test2017`` 里都已有文件时，不再下载、解压或写 CSV，也不打印。缺任何一个则下载四个 zip、
       解压并生成 CSV，终端打印 ``dataset_dir`` 以及压缩包、解压、CSV 三块。也可先按 README 手动解压，
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
      ``test.csv`` 仅 ``path_img``，来自 ``test-dev2017.txt``。test-dev 没有公开检测框，
      ``trainer.test`` 与 ``validate`` 一样要算 loss 和 mAP，实验里的 ``test_csv_paths`` 仍指向 ``val.csv``。
      ``predict.csv`` 仅 ``path_img``，来自验证集。``map_class_id_to_class_name.csv`` 为 COCO 80 类。

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
        """五个划分目录都有文件时什么都不打印。否则下载、解压并生成 CSV。

        Lightning 顺序：``prepare_data`` → ``setup``。写 CSV 与 ``datasets/COCO/coco_to_csv.py``
        共用 ``generate_coco_csvs``。
        """
        if coco_splits_ready(self.dataset_dir):
            return
        ensure_coco_files(self.dataset_dir)
        generate_coco_csvs(self.dataset_dir, overwrite=True, with_header=False)
