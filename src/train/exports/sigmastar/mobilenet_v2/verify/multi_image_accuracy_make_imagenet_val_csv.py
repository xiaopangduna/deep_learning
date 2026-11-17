#!/usr/bin/env python3
"""从已解压的 ILSVRC 验证集写 val.csv。

类别下标是 LOC_synset_mapping.txt 的行序，从 0 起，与 MobileNetV2 的 1000 类输出一致。
val.txt 的第二列是图片序号，不用。synset 取对应 XML 里第一个 object/name。
path_img 相对 CSV 所在目录。
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write ImageNet val.csv from an extracted ILSVRC tree")
    p.add_argument(
        "--root",
        required=True,
        help="Dataset root that contains ILSVRC/ and LOC_synset_mapping.txt",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV. Default is <root>/val.csv",
    )
    return p.parse_args()


def load_synset_index(path: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        synset = text.split(None, 1)[0]
        if synset in index:
            raise SystemExit(f"duplicate synset in {path}: {synset}")
        index[synset] = len(index)
    if len(index) != 1000:
        raise SystemExit(f"{path} has {len(index)} synsets, expect 1000")
    return index


def first_synset(xml_path: Path) -> str | None:
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return None
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is not None and name.text and name.text.strip():
            return name.text.strip()
    return None


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    mapping_path = root / "LOC_synset_mapping.txt"
    list_path = root / "ILSVRC/ImageSets/CLS-LOC/val.txt"
    image_dir = root / "ILSVRC/Data/CLS-LOC/val"
    ann_dir = root / "ILSVRC/Annotations/CLS-LOC/val"
    for path in (mapping_path, list_path, image_dir, ann_dir):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    synset_to_id = load_synset_index(mapping_path)
    out_path = Path(args.out) if args.out else root / "val.csv"
    rows: list[tuple[str, str, int]] = []
    skipped = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        stem = text.split(None, 1)[0]
        image_rel = f"ILSVRC/Data/CLS-LOC/val/{stem}.JPEG"
        if not (root / image_rel).is_file():
            print(f"missing image: {image_rel}", file=sys.stderr)
            skipped += 1
            continue
        synset = first_synset(ann_dir / f"{stem}.xml")
        if synset is None or synset not in synset_to_id:
            print(f"missing synset: {stem}", file=sys.stderr)
            skipped += 1
            continue
        rows.append((image_rel, synset, synset_to_id[synset]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_img", "class_name", "class_id"])
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)} skipped={skipped}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
