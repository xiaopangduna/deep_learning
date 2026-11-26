#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Google Drive 直链下载 CelebA 的对齐图像包和标注。

浏览器整包的下法见同目录 ``README.md``。这个脚本只下有稳定文件号的
7 个文件：对齐 JPEG 包，以及属性、身份、框、关键点和划分文本。
字节数和 MD5 都与官方包一致时跳过。半截文件按 HTTP Range 续传。

对齐图像包约 1.4GB。对它发不带 Range 的 GET，网盘会返回配额网页，
所以这里一律按区间拉。

示例::

    python datasets/CelebA/download_celeba.py
    python datasets/CelebA/download_celeba.py --output /data/celeba
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FOLDER_URL = (
    "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8"
    "?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg"
)
_CHUNK = 8 * 1024 * 1024
_DATASET_DIR = Path(__file__).resolve().parent / "celeba"


@dataclass(frozen=True)
class DriveFile:
    """一个有直链的官方文件。``relative`` 相对 ``celeba/``。"""

    relative: str
    file_id: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={self.file_id}&export=download&confirm=t"
        )


# 文件号、字节数、MD5 与 torchvision 记录的官方包一致，并与本目录里
# 浏览器整包解开后的成员对过。
FILES: tuple[DriveFile, ...] = (
    DriveFile(
        "Img/img_align_celeba.zip",
        "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        1_443_490_838,
        "00d2c5bc6d35e252742224ab0c1e8fcb",
    ),
    DriveFile(
        "Anno/list_attr_celeba.txt",
        "0B7EVK8r0v71pblRyaVFSWGxPY0U",
        26_721_026,
        "75e246fa4810816ffd6ee81facbd244c",
    ),
    DriveFile(
        "Anno/identity_CelebA.txt",
        "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
        3_424_458,
        "32bd1bd63d3c78cd57e08160ec5ed1e2",
    ),
    DriveFile(
        "Anno/list_bbox_celeba.txt",
        "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
        6_082_035,
        "00566efa6fedff7a56946cd1c10f1c16",
    ),
    DriveFile(
        "Anno/list_landmarks_align_celeba.txt",
        "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
        12_156_055,
        "cc24ecafdb5b50baae59b03474781f8c",
    ),
    DriveFile(
        "Anno/list_landmarks_celeba.txt",
        "0B7EVK8r0v71pTzJIdlJWdHczRlU",
        12_352_635,
        "063ee6ddb681f96bc9ca28c6febb9d1a",
    ),
    DriveFile(
        "Eval/list_eval_partition.txt",
        "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        2_836_386,
        "d32c9cbf5e040fd4025c592c306e6668",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CelebA 对齐图像包和标注")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DATASET_DIR,
        help="放置 Img/、Anno/、Eval/ 的目录",
    )
    return parser.parse_args()


def download_celeba(output: Path) -> None:
    """把 7 个官方文件下到 ``output``。已符合字节数和 MD5 的文件跳过。"""
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for spec in FILES:
        download_file(spec, root / spec.relative)


def download_file(spec: DriveFile, dest: Path) -> None:
    """下载一个文件。半截文件从已有字节续传，下完核对字节数和 MD5。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _matches(dest, spec):
        print(f"已在，跳过下载: {dest}")
        return

    existing = dest.stat().st_size if dest.is_file() else 0
    if existing >= spec.size:
        dest.unlink()
        existing = 0
    if existing:
        print(f"续传 {spec.relative}，已有 {existing} 字节: {dest}")
    else:
        print(f"下载 {spec.relative} 到 {dest}")

    next_percent = existing * 100 // spec.size // 10 * 10 + 10
    try:
        with dest.open("ab") as handle:
            while existing < spec.size:
                end = min(existing + _CHUNK, spec.size) - 1
                chunk = _fetch_range(spec.url, existing, end)
                expected = end - existing + 1
                if len(chunk) != expected:
                    raise RuntimeError(
                        f"{spec.relative} 区间长度不符: 收到 {len(chunk)} 字节，"
                        f"期望 {expected}"
                    )
                handle.write(chunk)
                existing += len(chunk)
                next_percent = _print_percent(
                    dest.name, existing, spec.size, next_percent
                )
    except Exception:
        if dest.is_file() and dest.stat().st_size == 0:
            dest.unlink()
        raise

    if not _matches(dest, spec):
        raise RuntimeError(
            f"{spec.relative} 与官方包不一致: {dest.stat().st_size} 字节，"
            f"MD5 {_md5(dest)}，期望 {spec.size} 字节、{spec.md5}"
        )


def _matches(path: Path, spec: DriveFile) -> bool:
    return path.is_file() and path.stat().st_size == spec.size and _md5(path) == spec.md5


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={start}-{end}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = getattr(response, "status", None) or response.getcode()
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"CelebA 下载失败: {url}") from exc
    if status != 206 or _looks_like_html(data):
        raise RuntimeError(
            "下载结果不是文件区间。网盘整段请求会返回配额网页，"
            f"请改下整个文件夹: {FOLDER_URL}"
        )
    return data


def _looks_like_html(data: bytes) -> bool:
    head = data.lstrip()[:20].lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_percent(name: str, done: int, total: int, next_percent: int) -> int:
    percent = done * 100 // total
    while percent >= next_percent and next_percent <= 100:
        print(f"  {name}  {next_percent}%")
        next_percent += 10
    return next_percent


def main() -> None:
    args = parse_args()
    download_celeba(args.output)


if __name__ == "__main__":
    main()
