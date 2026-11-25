"""LFWA 图像包下载，以及属性表转 CSV。

图像只走一条 ``lfw.tgz`` 直链。三个属性 ``.mat`` 在仓库的
``datasets/LFWA/lfw/``。DataModule 以后再接。
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
LFW_TRAIN_COUNT = 6_263

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
    attr_path: Path | None = None,
    overwrite: bool = False,
    expected_images: int = LFW_IMAGE_COUNT,
    train_count: int = LFW_TRAIN_COUNT,
) -> None:
    """把 ``list_attr_lfwa.txt`` 写成 ``train.csv`` 和 ``test.csv``。

    不读取 JPEG。原始值 ``1`` 写成 1，``-1`` 写成 0。扁平文件名
    ``George_HW_Bush_0001.jpg`` 写成 ``lfw/George_HW_Bush/George_HW_Bush_0001.jpg``，
    相对 ``dataset_dir``。前 ``train_count`` 行是训练集，其余是测试集。两个 CSV
    都在且 ``overwrite`` 为 False 时跳过写入，但仍检查属性文件。
    """
    root = _dataset_root(dataset_dir)
    source = Path(attr_path) if attr_path is not None else root / LFW_ATTR_NAME
    names, rows = parse_list_attr_lfwa(
        source, expected_images=expected_images, train_count=train_count
    )
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    if not overwrite and train_csv.exists() and test_csv.exists():
        print("LFWA CSV 已在，跳过生成:")
        print(f"  - {train_csv}")
        print(f"  - {test_csv}")
        return
    frame = pd.DataFrame(
        [
            {"path_img": image_relpath(filename), **dict(zip(names, values))}
            for filename, values in rows
        ],
        columns=["path_img", *names],
    )
    frame.iloc[:train_count].to_csv(train_csv, index=False)
    frame.iloc[train_count:].to_csv(test_csv, index=False)
    print(f"写出 {train_csv}，{train_count} 行")
    print(f"写出 {test_csv}，{len(frame) - train_count} 行")


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
