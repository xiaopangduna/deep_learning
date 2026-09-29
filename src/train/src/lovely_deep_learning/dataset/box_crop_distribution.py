# -*- encoding: utf-8 -*-
"""框级分类在筛选之后的分布。只读 CSV 列，不开图像。

丢掉的框按互斥顺序计数，三者之和加上保留数等于过滤前行数：

1. ``w`` 或 ``h`` 非正
2. 其余里短边 ``min(w*img_w, h*img_h)`` 小于 ``min_side_px``
3. 其余里类别不在映射或 ``class_groups`` 中

几何量只在保留下来的框上计算。建议损失权重为 ``n / (k * n_c)``，``k`` 是有样本的类别数，均值为 1。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

SHORT_SIDE_BINS = (
    ("lt_16", "16 以下", None, 16.0),
    ("16_32", "16-32", 16.0, 32.0),
    ("32_64", "32-64", 32.0, 64.0),
    ("64_128", "64-128", 64.0, 128.0),
    ("128_224", "128-224", 128.0, 224.0),
    ("ge_224", ">=224", 224.0, None),
)


def _parse_original_class_id(value: object) -> int:
    text = str(value).strip()
    if text == "":
        return -1
    return int(text)


def box_keep_masks(
    table: pd.DataFrame,
    *,
    min_side_px: float,
    has_label: bool,
    class_groups: Optional[dict[int, list[int]]],
    target_map: dict[int, str],
) -> dict[str, pd.Series]:
    """返回与 ``table`` 对齐的互斥布尔掩码：``invalid_wh``、``short_side``、``class_excluded``、``keep``。"""
    width = table["w"].astype(float)
    height = table["h"].astype(float)
    valid_wh = (width > 0) & (height > 0)
    invalid_wh = ~valid_wh
    if min_side_px > 0:
        short_side_px = np.minimum(
            width * table["img_w"].astype(float),
            height * table["img_h"].astype(float),
        )
        short_side = valid_wh & (short_side_px < min_side_px)
    else:
        short_side = pd.Series(False, index=table.index)
    remaining = valid_wh & ~short_side
    if has_label and class_groups:
        original_ids = table["class_id"].map(_parse_original_class_id)
        allowed = {
            original_id
            for members in class_groups.values()
            for original_id in members
        }
        class_excluded = remaining & ~original_ids.isin(allowed)
    elif has_label and target_map:
        names = set(target_map.values())
        class_excluded = remaining & ~table["class_name"].isin(names)
    else:
        class_excluded = pd.Series(False, index=table.index)
    keep = remaining & ~class_excluded
    return {
        "invalid_wh": invalid_wh,
        "short_side": short_side,
        "class_excluded": class_excluded,
        "keep": keep,
    }


def _percentiles(values: np.ndarray) -> dict[str, Optional[float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"p5": None, "p50": None, "p95": None}
    p5, p50, p95 = np.percentile(finite, [5, 50, 95])
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}


def _short_side_bins(short_side_px: np.ndarray) -> dict[str, int]:
    bins: dict[str, int] = {}
    for key, _label, lower, upper in SHORT_SIDE_BINS:
        chosen = np.ones(short_side_px.shape, dtype=bool)
        if lower is not None:
            chosen &= short_side_px >= lower
        if upper is not None:
            chosen &= short_side_px < upper
        bins[key] = int(chosen.sum())
    return bins


def geometry_summary(table: pd.DataFrame, box_scale: float) -> Optional[dict[str, Any]]:
    """保留框的短边、宽高比、面积，以及 ``box_scale`` 外扩后贴出图像边界的比例。"""
    if table.empty:
        return None
    img_w = table["img_w"].astype(float).to_numpy()
    img_h = table["img_h"].astype(float).to_numpy()
    box_w = table["w"].astype(float).to_numpy() * img_w
    box_h = table["h"].astype(float).to_numpy() * img_h
    short_side_px = np.minimum(box_w, box_h)
    area_px = box_w * box_h
    aspect = np.divide(box_w, box_h, out=np.full_like(box_w, np.nan), where=box_h > 0)
    abs_w = np.maximum(box_w * box_scale, 1.0)
    abs_h = np.maximum(box_h * box_scale, 1.0)
    abs_cx = table["cx"].astype(float).to_numpy() * img_w
    abs_cy = table["cy"].astype(float).to_numpy() * img_h
    x1 = np.floor(abs_cx - abs_w / 2.0)
    y1 = np.floor(abs_cy - abs_h / 2.0)
    x2 = np.ceil(abs_cx + abs_w / 2.0)
    y2 = np.ceil(abs_cy + abs_h / 2.0)
    clipped = (x1 < 0) | (y1 < 0) | (x2 > img_w) | (y2 > img_h)
    return {
        "short_side_px": _percentiles(short_side_px),
        "short_side_bins": _short_side_bins(short_side_px),
        "aspect": _percentiles(aspect),
        "area_px": _percentiles(area_px),
        "clip_ratio": float(clipped.mean()),
    }


def class_summary(
    table: pd.DataFrame,
    class_map: dict[int, str],
    member_counts: Optional[dict[int, list[dict[str, Any]]]] = None,
) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    """过滤后的类别计数、占比，以及有样本类别上的逆频率权重。"""
    observed: dict[int, int] = {}
    observed_name: dict[int, str] = {}
    if not table.empty and "class_id" in table.columns:
        class_ids = table["class_id"].map(_parse_original_class_id)
        for class_id, class_name in zip(class_ids, table["class_name"].astype(str)):
            if class_id < 0:
                continue
            observed[class_id] = observed.get(class_id, 0) + 1
            observed_name.setdefault(class_id, class_name)
    all_ids = set(class_map) | set(observed)
    n_kept = sum(observed.values())
    positive_ids = [class_id for class_id in sorted(all_ids) if observed.get(class_id, 0) > 0]
    k = len(positive_ids)
    classes: list[dict[str, Any]] = []
    for class_id in sorted(all_ids):
        count = observed.get(class_id, 0)
        weight = None if count == 0 or k == 0 else float(n_kept) / (k * count)
        entry: dict[str, Any] = {
            "class_id": int(class_id),
            "class_name": class_map.get(class_id, observed_name.get(class_id, str(class_id))),
            "count": int(count),
            "ratio": 0.0 if n_kept == 0 else float(count) / n_kept,
            "weight": weight,
        }
        if member_counts is not None and class_id in member_counts:
            entry["members"] = member_counts[class_id]
        classes.append(entry)
    weights = [float(entry["weight"]) for entry in classes if entry["count"] > 0]
    weight_ids = [int(entry["class_id"]) for entry in classes if entry["count"] > 0]
    return classes, weights, weight_ids


def mapping_status(table: pd.DataFrame, class_map: dict[int, str]) -> dict[str, Any]:
    """类别映射是否与过滤后的表一致，以及 ``class_id`` 是否连续。只记录，不中断训练。"""
    warnings: list[str] = []
    if table.empty or "class_id" not in table.columns:
        return {"consistent": True, "continuous": True, "span": None, "warnings": warnings}
    raw = table["class_id"].astype(str).str.strip()
    numeric = pd.to_numeric(raw.where(raw != "", pd.NA), errors="coerce")
    bad_mask = numeric.isna()
    if bad_mask.any():
        bad_values = sorted(set(table.loc[bad_mask, "class_id"].astype(str)))
        warnings.append(f"发现无法转换为整数的 class_id: {bad_values}")
    valid = table.loc[~bad_mask]
    valid_ids = numeric.loc[~bad_mask].astype(int)
    consistent = not bool(bad_mask.any())
    actual = set(int(value) for value in valid_ids.tolist())
    if class_map:
        expected = valid_ids.map(class_map)
        names = valid["class_name"].astype(str)
        mismatch = expected.isna() | (expected.astype(str) != names)
        if bool(mismatch.any()):
            consistent = False
            shown = valid.loc[mismatch.to_numpy(), ["class_id", "class_name"]].drop_duplicates().head(8)
            for class_id, class_name in shown.itertuples(index=False):
                parsed = _parse_original_class_id(class_id)
                mapped = class_map.get(parsed, "（无）")
                warnings.append(
                    f"map_class_id_to_class_name 与实际数据不一致: ID {parsed}, "
                    f"映射中为 {mapped!r}, 实际为 {class_name!r}"
                )
        unused = sorted(set(class_map) - actual)
        if unused:
            warnings.append(f"映射中存在数据中未使用的 ID: {unused}")
    else:
        warnings.append("map_class_id_to_class_name 为空，跳过与映射表的校验")
    continuous = True
    span = None
    if actual:
        min_id, max_id = min(actual), max(actual)
        missing = sorted(set(range(min_id, max_id + 1)) - actual)
        if missing:
            continuous = False
            warnings.append(f"class_id 不连续，缺少 ID: {missing}")
        else:
            span = str(min_id) if min_id == max_id else f"{min_id}-{max_id}"
    return {
        "consistent": consistent,
        "continuous": continuous,
        "span": span,
        "warnings": warnings,
    }


def assemble_distribution(
    *,
    n_raw: int,
    masks: dict[str, pd.Series],
    kept: pd.DataFrame,
    box_scale: float,
    class_map: dict[int, str],
    member_counts: Optional[dict[int, list[dict[str, Any]]]],
    has_label: bool,
) -> dict[str, Any]:
    dropped = {
        "invalid_wh": int(masks["invalid_wh"].sum()),
        "short_side": int(masks["short_side"].sum()),
        "class_excluded": int(masks["class_excluded"].sum()),
    }
    n_kept = int(len(kept))
    n_dropped = sum(dropped.values())
    if n_raw != n_dropped + n_kept:
        raise RuntimeError(
            f"过滤计数对不上: 过滤前 {n_raw}，丢掉 {n_dropped}，过滤后 {n_kept}"
        )
    if has_label:
        classes, weights, weight_ids = class_summary(kept, class_map, member_counts)
        mapping = mapping_status(kept, class_map)
    else:
        classes, weights, weight_ids = [], [], []
        mapping = {"consistent": True, "continuous": True, "span": None, "warnings": []}
    n_images = int(kept["img_path"].astype(str).nunique()) if n_kept else 0
    return {
        "n_raw": int(n_raw),
        "dropped": dropped,
        "n_dropped": int(n_dropped),
        "n_kept": n_kept,
        "n_images": n_images,
        "boxes_per_image": None if n_images == 0 else float(n_kept) / n_images,
        "classes": classes,
        "suggested_weight": weights,
        "suggested_weight_class_ids": weight_ids,
        "geometry": geometry_summary(kept, box_scale),
        "mapping": mapping,
    }


def training_class_rows(classes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """训练日志用的类别行：融合后的 id、名称、留下的框数和占比。"""
    return [
        {
            "class_id": int(entry["class_id"]),
            "class_name": str(entry["class_name"]),
            "count": int(entry["count"]),
            "ratio": float(entry["ratio"]),
        }
        for entry in classes
    ]


def _class_table(classes: Sequence[dict[str, Any]]) -> list[str]:
    if not classes:
        return ["类别: （无）"]
    names = [str(entry["class_name"]) for entry in classes]
    name_width = max(len("class_name"), max(len(name) for name in names))
    lines = [f"{'class_id':>8}  {'class_name':<{name_width}}  {'count':>7}  {'ratio':>6}"]
    for entry in classes:
        lines.append(
            f"{entry['class_id']:>8}  {entry['class_name']:<{name_width}}  "
            f"{entry['count']:>7}  {entry['ratio']:>6.3f}"
        )
    return lines


def _ruled_block(lines: list[str]) -> str:
    width = max([76, *(len(line) for line in lines)])
    bar = "=" * width
    rule = "-" * width
    title, *rest = lines
    return "\n".join([bar, title, rule, *rest, bar])


def format_distribution_report(payload: dict[str, Any]) -> str:
    """训练和验证各一张类别表。行是筛完并完成类融合之后的训练类。"""
    blocks: list[str] = []
    for split in payload["splits"]:
        paths = ", ".join(str(path) for path in split["csv_paths"])
        lines = [f"{split['title']}  {paths}"]
        lines.extend(_class_table(split["classes"]))
        lines.extend(split.get("warnings") or [])
        blocks.append(_ruled_block(lines))
    return "\n".join(blocks)
