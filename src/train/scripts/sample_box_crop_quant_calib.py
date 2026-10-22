#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从框级 CSV 随机抽样，裁框 + LetterBox，写出量化校准图。

默认对齐 ``image_classifiter_from_detect_box_head_eye_open`` 实验的 val 预处理：
``box_scale=1.0``、``min_side_px=128``、``LetterBox(224, fill=114)``。
只写 uint8 图像（不做 Normalize），供板端校准工具读取。

示例::

    uv run scripts/sample_box_crop_quant_calib.py \\
        --csv datasets/head_eye_open/train_v301.csv \\
        --out-dir exports/quant_data/head_eye_open_v301_calib500 \\
        --num 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.tv_tensors import Image as TvImage

from lovely_deep_learning.dataset.base import BaseDataset
from lovely_deep_learning.dataset.box_crop_distribution import box_keep_masks
from lovely_deep_learning.dataset.image_classifier import ImageClassifierFromDetectBoxDataset
from lovely_deep_learning.transforms.letterbox import LetterBox

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="随机抽样检测框，裁切并 LetterBox，生成量化校准图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "datasets/head_eye_open/train_v301.csv",
        help="框级 CSV（path_img,cx,cy,w,h,img_w,img_h[,class_*]）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "exports/quant_data/head_eye_open_v301_calib500",
        help="输出目录（图像 + dataset.txt + manifest.csv）",
    )
    p.add_argument("--num", type=int, default=500, help="抽样数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--box-scale", type=float, default=1.0, help="裁框外扩倍率")
    p.add_argument("--min-side-px", type=float, default=128.0, help="短边过滤阈值（像素）")
    p.add_argument("--size", type=int, default=224, help="LetterBox 边长（正方形）")
    p.add_argument("--fill", type=int, default=114, help="LetterBox 填充值（uint8）")
    p.add_argument(
        "--no-scaleup",
        action="store_true",
        help="LetterBox 禁止放大（默认与训练一致允许放大）",
    )
    p.add_argument(
        "--ext",
        choices=("jpg", "png"),
        default="jpg",
        help="输出图像格式",
    )
    p.add_argument(
        "--stratify",
        action="store_true",
        help="按 class_id 分层抽样（无类别列时忽略）",
    )
    return p.parse_args()


def _load_and_filter(csv_path: Path, min_side_px: float) -> pd.DataFrame:
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
    required = ("path_img", "cx", "cy", "w", "h", "img_w", "img_h")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列 {missing}")
    BaseDataset.resolve_relative_path_columns_if_needed(df, csv_path.parent, tuple(df.columns))
    work = df.rename(columns={"path_img": "img_path"})
    for col in ("cx", "cy", "w", "h", "img_w", "img_h"):
        work[col] = work[col].astype(float)
    has_label = "class_id" in work.columns
    masks = box_keep_masks(
        work,
        min_side_px=min_side_px,
        has_label=has_label,
        class_groups=None,
    )
    kept = work.loc[masks["keep"]].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError(f"筛选后无样本（min_side_px={min_side_px}）")
    return kept


def _sample_rows(df: pd.DataFrame, num: int, seed: int, stratify: bool) -> pd.DataFrame:
    if num <= 0:
        raise ValueError(f"--num 须为正整数，收到 {num}")
    if num > len(df):
        raise ValueError(f"可用样本 {len(df)} 少于请求数量 {num}")
    rng = np.random.default_rng(seed)
    if stratify and "class_id" in df.columns:
        groups = list(df.groupby("class_id", sort=True))
        n_groups = len(groups)
        base, rem = divmod(num, n_groups)
        quotas = {gid: base + (1 if i < rem else 0) for i, (gid, _) in enumerate(groups)}
        parts: list[pd.DataFrame] = []
        leftover_need = 0
        for gid, g in groups:
            q = quotas[gid]
            if len(g) >= q:
                idx = rng.choice(len(g), size=q, replace=False)
                parts.append(g.iloc[idx])
            else:
                parts.append(g)
                leftover_need += q - len(g)
        sampled = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
        if leftover_need > 0:
            used = set(map(tuple, sampled[["img_path", "cx", "cy", "w", "h"]].itertuples(index=False)))
            pool = df[
                ~df.apply(
                    lambda r: (r["img_path"], r["cx"], r["cy"], r["w"], r["h"]) in used,
                    axis=1,
                )
            ]
            if len(pool) < leftover_need:
                raise ValueError("分层抽样补齐失败：剩余池不足")
            idx = rng.choice(len(pool), size=leftover_need, replace=False)
            sampled = pd.concat([sampled, pool.iloc[idx]], ignore_index=True)
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    idx = rng.choice(len(df), size=num, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _letterbox_crop(
    crop_bgr: np.ndarray,
    letterbox: LetterBox,
) -> np.ndarray:
    """BGR crop → LetterBox → BGR ``(H, W, 3)`` uint8。"""
    rgb = BaseDataset.convert_img_from_numpy_to_tensor_uint8(crop_bgr)
    out = letterbox(TvImage(rgb))
    if not torch.is_tensor(out):
        raise TypeError(f"LetterBox 输出类型异常: {type(out)}")
    return BaseDataset.convert_img_from_tensor_to_numpy_uint8(out)


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else (ROOT / args.csv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = _load_and_filter(csv_path, args.min_side_px)
    sampled = _sample_rows(table, args.num, args.seed, args.stratify)

    letterbox = LetterBox(
        size=[args.size, args.size],
        fill=args.fill,
        center=True,
        scaleup=not args.no_scaleup,
        interpolation=2,
    )

    names: list[str] = []
    manifest_rows: list[dict] = []
    skipped = 0
    for i, row in sampled.iterrows():
        img_path = str(row["img_path"])
        img_np, _ = BaseDataset.read_img(img_path, None)
        if img_np is None:
            skipped += 1
            print(f"[skip] 无法读取: {img_path}")
            continue
        crop = ImageClassifierFromDetectBoxDataset.crop_cxcywh(
            img_np,
            float(row["cx"]),
            float(row["cy"]),
            float(row["w"]),
            float(row["h"]),
            args.box_scale,
        )
        lb_bgr = _letterbox_crop(crop, letterbox)
        name = f"{i:05d}.{args.ext}"
        save_path = out_dir / name
        if args.ext == "jpg":
            ok = cv2.imwrite(str(save_path), lb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            ok = cv2.imwrite(str(save_path), lb_bgr)
        if not ok:
            skipped += 1
            print(f"[skip] 写入失败: {save_path}")
            continue
        names.append(name)
        rec = {
            "file": name,
            "img_path": img_path,
            "cx": float(row["cx"]),
            "cy": float(row["cy"]),
            "w": float(row["w"]),
            "h": float(row["h"]),
            "img_w": float(row["img_w"]),
            "img_h": float(row["img_h"]),
        }
        if "class_id" in row.index and str(row["class_id"]).strip() != "":
            rec["class_id"] = int(float(row["class_id"]))
        if "class_name" in row.index:
            rec["class_name"] = str(row["class_name"])
        manifest_rows.append(rec)

    if not names:
        raise RuntimeError("没有成功写出任何校准图")

    dataset_txt = out_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(names) + "\n", encoding="utf-8")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    meta = {
        "csv": str(csv_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_requested": args.num,
        "num_written": len(names),
        "num_skipped": skipped,
        "pool_after_filter": len(table),
        "seed": args.seed,
        "box_scale": args.box_scale,
        "min_side_px": args.min_side_px,
        "letterbox": {
            "size": [args.size, args.size],
            "fill": args.fill,
            "center": True,
            "scaleup": not args.no_scaleup,
            "interpolation": 2,
        },
        "stratify": bool(args.stratify),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"写出 {len(names)} 张校准图 → {out_dir}\n"
        f"  dataset.txt / manifest.csv / meta.json\n"
        f"  筛选后池={len(table)}, 跳过={skipped}"
    )


if __name__ == "__main__":
    main()
