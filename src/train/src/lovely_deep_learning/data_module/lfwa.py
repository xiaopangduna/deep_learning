"""LFWA 图像包下载，以及属性 ``.mat`` 转 CSV。

图像只走一条 ``lfw.tgz`` 直链。三个属性 ``.mat`` 在仓库的
``datasets/LFWA/lfw/``。40 个属性和 73 个属性各写一套 CSV。DataModule 以后再接。
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LFW_GDRIVE_PAGE = (
    "https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing"
)
LFW_TGZ_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t"
)
LFW_ARCHIVE_NAME = "lfw.tgz"
LFW_ARCHIVE_BYTES = 180_566_744
LFW_IMAGE_COUNT = 13_233
LFW_IDENTITY_COUNT = 5_749
LFW_IMAGE_ROOT = "lfw"
LFW_ATTR_NAME = "list_attr_lfwa.txt"
LFW_ATTR_FOLDER = (
    "https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams"
    "?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA"
)
LFW_ATTR_COUNT = 40
LFW_ATTR73_COUNT = 73
LFW_TRAIN_COUNT = 6_263
LFW_ATT40_NAME = "lfw_att_40.mat"
LFW_ATT73_NAME = "lfw_att_73.mat"
LFW_SPLIT_NAME = "indices_train_test.mat"
LFW_VAL_IMAGE_TARGET = 626
LFW_VAL_SEED = 0
LFW_CSV_SPLITS = ("train", "val", "test")

_LABEL_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".xml"}
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class LfwArchiveInfo:
    """压缩包里数出来的图像、身份和标注文件。"""

    images: int
    identities: int
    label_names: tuple[str, ...]


def inspect_lfw_tgz(archive: Path) -> LfwArchiveInfo:
    """统计 gzip tar 里的 JPEG、身份目录和标注文件。不把内容解出来。"""
    images = 0
    identities: set[str] = set()
    labels: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            suffix = Path(name).suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                images += 1
                parts = Path(name).parts
                if len(parts) >= 3 and parts[0] == LFW_IMAGE_ROOT:
                    identities.add(parts[1])
            elif suffix in _LABEL_SUFFIXES:
                labels.append(name)
    return LfwArchiveInfo(images, len(identities), tuple(labels))


def download_lfw_tgz(dest: Path, url: str = LFW_TGZ_URL) -> None:
    """下载 ``lfw.tgz``。大小已经是官方包时跳过。半截文件按 Range 续传。

    ``url`` 默认是网盘文件的直链。分享页会返回 HTML 确认页，不能直接当压缩包下。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == LFW_ARCHIVE_BYTES:
        print(f"LFW 图像包已在，跳过下载: {dest}")
        return

    existing = dest.stat().st_size if dest.is_file() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if 0 < existing < LFW_ARCHIVE_BYTES:
        headers["Range"] = f"bytes={existing}-"
        print(f"续传 LFW 图像包，已有 {existing} 字节: {dest}")
    else:
        existing = 0
        print(f"下载 LFW 图像包到 {dest}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None) or response.getcode()
        appending = bool(existing and status == 206)
        mode = "ab" if appending else "wb"
        done = existing if appending else 0
        next_percent = 10
        first = response.read(_CHUNK)
        if not first:
            raise RuntimeError(f"LFW 图像包下载为空: {url}")
        if not appending and not first.startswith(b"\x1f\x8b"):
            raise RuntimeError(
                "下载结果不是 gzip。分享页返回的是确认页，请用直链 "
                f"{LFW_TGZ_URL}"
            )
        with dest.open(mode) as handle:
            handle.write(first)
            done += len(first)
            next_percent = _print_percent(dest.name, done, next_percent)
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                next_percent = _print_percent(dest.name, done, next_percent)

    size = dest.stat().st_size
    if size != LFW_ARCHIVE_BYTES:
        raise RuntimeError(
            f"LFW 图像包大小不符: {size} 字节，期望 {LFW_ARCHIVE_BYTES}"
        )


def extract_lfw_tgz(archive: Path, dataset_dir: Path) -> None:
    """把 ``lfw/`` 解到 ``dataset_dir``。成员路径不能逃出这个目录。"""
    root = dataset_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            relative = member.name.rstrip("/")
            if relative in ("", "."):
                continue
            dest = _safe_dest(root, relative)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"LFW 图像包读不出成员: {member.name}")
            with source, dest.open("wb") as handle:
                while True:
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)


def prepare_lfw_images(dataset_dir, *, url: str = LFW_TGZ_URL) -> LfwArchiveInfo:
    """保证 ``dataset_dir/lfw.tgz`` 和 ``dataset_dir/lfw/`` 就位。

    图像张数或身份数对不上就停下，不把残包解进去。包里没有属性文件时只打印说明，
    不把它当成失败：这份发布物本身就没有标签。
    """
    root = _dataset_root(dataset_dir)
    archive = root / LFW_ARCHIVE_NAME
    download_lfw_tgz(archive, url)
    info = inspect_lfw_tgz(archive)
    if info.images != LFW_IMAGE_COUNT or info.identities != LFW_IDENTITY_COUNT:
        raise RuntimeError(
            "LFW 图像包内容不符: "
            f"{info.images} 张图像、{info.identities} 个身份，"
            f"期望 {LFW_IMAGE_COUNT} 张、{LFW_IDENTITY_COUNT} 个身份"
        )
    if _images_ready(root):
        print(f"LFW 图像已解压，跳过: {root / LFW_IMAGE_ROOT}")
    else:
        print(f"解压 {archive}")
        extract_lfw_tgz(archive, root)
        if not _images_ready(root):
            raise RuntimeError(f"解压后不足 {LFW_IMAGE_COUNT} 张 JPEG: {root / LFW_IMAGE_ROOT}")
    _print_label_status(info)
    return info


def generate_lfwa_csvs(
    dataset_dir,
    *,
    overwrite: bool = False,
    val_image_target: int = LFW_VAL_IMAGE_TARGET,
    val_seed: int = LFW_VAL_SEED,
) -> None:
    """从三个 ``.mat`` 写出 40 属性与 73 属性的 ``train`` / ``val`` / ``test`` CSV。

    两套表的 ``path_img`` 和框列相同，属性列分开。40 与 73 有 38 个名字在加上
    ``attr_`` 前缀后会撞车，所以不写进同一个文件。划分用
    ``indices_train_test.mat`` 的图像下标。测试集原样留下。训练集按身份留出
    验证集：身份名字排序后用 ``RandomState(val_seed)`` 打乱，按这个顺序加人，
    加到图像数达到 ``val_image_target`` 为止。同一个人只进入一个文件。

    ``path_img`` 相对 ``dataset_dir``。LFWA 的图已经是脸，框是整张图。
    六个 CSV 都在且 ``overwrite`` 为 False 时跳过写入，但仍检查三个 ``.mat``。
    """
    root = _dataset_root(dataset_dir)
    outputs = _csv_paths(root)
    names40, paths40, label40 = load_attribute_mat(root / LFW_ATT40_NAME, LFW_ATTR_COUNT)
    names73, paths73, label73 = load_attribute_mat(root / LFW_ATT73_NAME, LFW_ATTR73_COUNT)
    if paths40 != paths73:
        raise RuntimeError("lfw_att_40.mat 与 lfw_att_73.mat 的图像顺序不一致")
    train_index, test_index = load_split_mat(root / LFW_SPLIT_NAME, image_count=len(paths40))
    if not overwrite and all(path.is_file() for path in outputs.values()):
        print("LFWA CSV 已在，跳过生成:")
        for path in outputs.values():
            print(f"  - {path}")
        return

    train_kept, val_index = split_train_val(
        paths40, train_index, val_image_target=val_image_target, seed=val_seed
    )
    sizes = [jpeg_size(root / relative) for relative in paths40]
    splits = {"train": train_kept, "val": val_index, "test": test_index}
    for count, names, label in (
        (LFW_ATTR_COUNT, names40, label40),
        (LFW_ATTR73_COUNT, names73, label73),
    ):
        columns = [attribute_column(name) for name in names]
        if len(columns) != len(set(columns)):
            raise RuntimeError(f"{count} 个属性列名重复")
        for stem, index in splits.items():
            dest = outputs[f"{stem}_{count}"]
            _write_attribute_csv(dest, paths40, sizes, columns, label, index)
            print(f"写出 {dest}，{len(index)} 行")


def load_attribute_mat(path: Path, attr_count: int) -> tuple[list[str], list[str], np.ndarray]:
    """读取属性 ``.mat``。返回属性名、``path_img`` 列表、形状 ``(属性, 图像)`` 的 0/1 标签。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        names = _matlab_strings(handle, "AttrName")
        stored = _matlab_strings(handle, "name")
        label = np.asarray(handle["label"])
    if len(names) != attr_count:
        raise RuntimeError(f"{path} 应有 {attr_count} 个属性，收到 {len(names)} 个")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path} 属性名重复")
    if label.shape != (attr_count, len(stored)):
        raise RuntimeError(f"{path} 标签形状是 {label.shape}，期望 {(attr_count, len(stored))}")
    if not np.isin(label, (0, 1)).all():
        raise RuntimeError(f"{path} 标签只能是 0 或 1")
    paths = [mat_image_relpath(name) for name in stored]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{path} 图像路径重复")
    return names, paths, label.astype(np.int64, copy=False)


def load_split_mat(path: Path, *, image_count: int) -> tuple[np.ndarray, np.ndarray]:
    """读取图像划分。返回从 0 开始、按 ``.mat`` 顺序排列的训练和测试下标。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        train = _matlab_indices(handle["indices_img_train"], image_count, "indices_img_train")
        test = _matlab_indices(handle["indices_img_test"], image_count, "indices_img_test")
    if set(train.tolist()) & set(test.tolist()):
        raise RuntimeError(f"{path} 训练和测试图像下标重叠")
    covered = np.concatenate([train, test])
    if covered.size != image_count or set(covered.tolist()) != set(range(image_count)):
        raise RuntimeError(f"{path} 图像下标没有盖住全部 {image_count} 张有标签的图")
    return train, test


def split_train_val(
    paths: list[str],
    train_index: np.ndarray,
    *,
    val_image_target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从训练下标里按人留出验证集。返回仍按原下标顺序排列的训练、验证下标。"""
    if val_image_target <= 0 or val_image_target >= len(train_index):
        raise RuntimeError(
            f"验证图像数 {val_image_target} 应在 1 到 {len(train_index) - 1} 之间"
        )
    grouped: dict[str, list[int]] = {}
    for index in train_index.tolist():
        grouped.setdefault(_identity(paths[index]), []).append(index)
    order = np.random.RandomState(seed).permutation(len(grouped))
    chosen: list[int] = []
    for position in order:
        if len(chosen) >= val_image_target:
            break
        chosen.extend(grouped[sorted(grouped)[int(position)]])
    val_set = set(chosen)
    if not val_set or len(val_set) == len(train_index):
        raise RuntimeError("验证集留空了训练集或验证集")
    val_index = np.array([index for index in train_index.tolist() if index in val_set], dtype=np.int64)
    kept = np.array([index for index in train_index.tolist() if index not in val_set], dtype=np.int64)
    return kept, val_index


def attribute_column(name: str) -> str:
    """``Middle Aged`` → ``attr_Middle_Aged``。空白换成下划线，其余字符保留。"""
    column = "attr_" + "_".join(name.split())
    if column == "attr_":
        raise RuntimeError(f"属性名为空: {name!r}")
    return column


def mat_image_relpath(stored: str) -> str:
    """``Aaron_Eckhart\\Aaron_Eckhart_0001.jpg`` → ``lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg``。"""
    normalized = stored.replace("\\", "/")
    filename = Path(normalized).name
    relative = image_relpath(filename)
    folder = Path(normalized).parent.name
    if folder and folder != _identity(relative):
        raise RuntimeError(f"属性表里的目录和文件名对不上: {stored}")
    return relative


def jpeg_size(path: Path) -> tuple[int, int]:
    """读 JPEG 的 SOF 宽高，不解码像素。返回 ``(宽, 高)``。"""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise RuntimeError(f"不是 JPEG: {path}")
        while True:
            if handle.read(1) != b"\xff":
                raise RuntimeError(f"JPEG 标记损坏: {path}")
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker == b"":
                raise RuntimeError(f"JPEG 读不到宽高: {path}")
            code = marker[0]
            if code == 0x01 or code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) < 2:
                raise RuntimeError(f"JPEG 读不到宽高: {path}")
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise RuntimeError(f"JPEG 标记损坏: {path}")
            if code in (0xC0, 0xC1, 0xC2):
                payload = handle.read(5)
                if len(payload) < 5:
                    raise RuntimeError(f"JPEG 读不到宽高: {path}")
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"JPEG 宽高非法: {path}")
                return width, height
            handle.seek(length - 2, 1)


def parse_list_attr_lfwa(
    path: Path,
    *,
    expected_images: int = LFW_IMAGE_COUNT,
    train_count: int = LFW_TRAIN_COUNT,
) -> tuple[list[str], list[tuple[str, list[int]]]]:
    """读取 LFWA 属性表。返回属性名，以及 ``(文件名, 0/1 列表)``。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 {path}。{LFW_ATTR_NAME} 不在 lfw.tgz 里，"
            f"官方目录 {LFW_ATTR_FOLDER} 里是 lfw_att_40.mat，没有这份 txt。"
            "手动下载见 datasets/LFWA/README.md。"
        )
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"{path} 至少要有图像数、属性名和一行标注")
    try:
        declared = int(lines[0])
    except ValueError as exc:
        raise RuntimeError(f"{path} 第一行应是图像数，收到 {lines[0]!r}") from exc
    names = lines[1].split()
    if len(names) != LFW_ATTR_COUNT:
        raise RuntimeError(f"{path} 应有 {LFW_ATTR_COUNT} 个属性，收到 {len(names)} 个")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path} 属性名重复")
    rows: list[tuple[str, list[int]]] = []
    for line_no, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) != 1 + LFW_ATTR_COUNT:
            raise RuntimeError(f"{path}:{line_no} 应有文件名和 {LFW_ATTR_COUNT} 个值")
        values: list[int] = []
        for raw in parts[1:]:
            if raw == "1":
                values.append(1)
            elif raw == "-1":
                values.append(0)
            else:
                raise RuntimeError(f"{path}:{line_no} 属性值只能是 1 或 -1，收到 {raw}")
        rows.append((parts[0], values))
    if declared != len(rows):
        raise RuntimeError(f"{path} 声明 {declared} 行，实际 {len(rows)} 行")
    if expected_images is not None and len(rows) != expected_images:
        raise RuntimeError(f"{path} 有 {len(rows)} 行，期望 {expected_images} 行")
    if not 0 < train_count < len(rows):
        raise RuntimeError(f"训练行数 {train_count} 不在 1 到 {len(rows) - 1} 之间")
    return names, rows


def image_relpath(filename: str) -> str:
    """``George_HW_Bush_0001.jpg`` → ``lfw/George_HW_Bush/George_HW_Bush_0001.jpg``。"""
    name = Path(filename).name
    if not name.lower().endswith(".jpg"):
        raise RuntimeError(f"LFWA 文件名应以 .jpg 结尾: {filename}")
    identity, separator, number = name[:-4].rpartition("_")
    if not separator or not identity or len(number) != 4 or not number.isdigit():
        raise RuntimeError(f"LFWA 文件名应为 <身份>_0001.jpg: {filename}")
    return f"{LFW_IMAGE_ROOT}/{identity}/{name}"


def _csv_paths(root: Path) -> dict[str, Path]:
    paths = {}
    for count in (LFW_ATTR_COUNT, LFW_ATTR73_COUNT):
        for stem in LFW_CSV_SPLITS:
            paths[f"{stem}_{count}"] = root / f"{stem}_{count}.csv"
    return paths


def _write_attribute_csv(
    dest: Path,
    paths: list[str],
    sizes: list[tuple[int, int]],
    columns: list[str],
    label: np.ndarray,
    index: np.ndarray,
) -> None:
    chosen = index.tolist()
    frame = pd.DataFrame(
        {
            "path_img": [paths[i] for i in chosen],
            "cx": "0.5",
            "cy": "0.5",
            "w": "1",
            "h": "1",
            "img_w": [sizes[i][0] for i in chosen],
            "img_h": [sizes[i][1] for i in chosen],
        }
    )
    for column_index, column in enumerate(columns):
        frame[column] = label[column_index, index]
    frame.to_csv(dest, index=False)


def _matlab_strings(handle, key: str) -> list[str]:
    if key not in handle:
        raise RuntimeError(f"MATLAB 文件缺少 {key}")
    values = []
    for ref in handle[key][()].reshape(-1):
        chars = handle[ref][()].reshape(-1)
        values.append("".join(chr(int(char)) for char in chars))
    return values


def _matlab_indices(dataset, image_count: int, key: str) -> np.ndarray:
    raw = np.asarray(dataset).reshape(-1)
    if raw.size == 0 or not np.equal(raw, np.floor(raw)).all():
        raise RuntimeError(f"{key} 应是正整数下标")
    index = raw.astype(np.int64) - 1
    if (index < 0).any() or (index >= image_count).any() or len(set(index.tolist())) != index.size:
        raise RuntimeError(f"{key} 下标超出 1 到 {image_count} 或有重复")
    return index


def _identity(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) != 3 or parts[0] != LFW_IMAGE_ROOT:
        raise RuntimeError(f"图像路径应为 lfw/<身份>/<文件名>: {relative}")
    return parts[1]


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "读取 MATLAB 7.3 需要 h5py。在 src/train 下执行 "
            "uv run --with h5py python datasets/LFWA/lfwa_attr_to_csv.py"
        ) from exc
    return h5py


def _dataset_root(dataset_dir) -> Path:
    path = Path(dataset_dir).expanduser()
    if path.is_symlink() and not path.exists():
        raise FileNotFoundError(f"LFW dataset_dir 不存在: {path}")
    path.mkdir(parents=True, exist_ok=True)
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LFW dataset_dir 不存在: {root}")
    return root


def _images_ready(root: Path) -> bool:
    image_root = root / LFW_IMAGE_ROOT
    if not image_root.is_dir():
        return False
    count = 0
    for identity in image_root.iterdir():
        if not identity.is_dir():
            continue
        count += sum(
            1
            for path in identity.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
    return count == LFW_IMAGE_COUNT


def _print_label_status(info: LfwArchiveInfo) -> None:
    print("=" * 76)
    if info.label_names:
        print("LFW 图像包里有这些标注文件:")
        for name in info.label_names:
            print(f"  - {name}")
    else:
        print("LFW 图像包里没有属性标签。")
        print("LFWA 的 40 个二值属性在单独的 list_attr_lfwa.txt，这份 tgz 不含该文件。")
    print("=" * 76)


def _print_percent(name: str, done: int, next_percent: int) -> int:
    percent = done * 100 // LFW_ARCHIVE_BYTES
    while percent >= next_percent and next_percent <= 100:
        print(f"  {name}  {next_percent}%")
        next_percent += 10
    return next_percent


def _safe_dest(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"LFW 图像包路径非法: {relative}")
    dest = (root / relative_path).resolve()
    if not dest.is_relative_to(root):
        raise RuntimeError(f"LFW 图像包路径非法: {relative}")
    return dest
