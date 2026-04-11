"""通用 I/O：表格导出、实例列表 JSON 序列化等。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _class_display_name_or_id(
    class_names: Mapping[Any, str] | None, class_id: int
) -> str:
    """有映射则用类名字符串，否则用 ``str(class_id)``（键支持 int 或 str）。"""
    if class_names is not None:
        name = class_names.get(class_id, class_names.get(str(class_id)))
        if name is not None:
            return name
    return str(class_id)


def instances_to_json_str(
    xyxy: np.ndarray,
    cls_ids: np.ndarray,
    class_names: Mapping[Any, str] | None,
    scores: np.ndarray | None = None,
) -> str:
    """每张图的目标列表，写入 CSV 的一列（JSON 文本）；``class_name`` 恒为可读标签。"""
    rows: list[dict[str, Any]] = []
    for j in range(xyxy.shape[0]):
        cid = int(cls_ids[j])
        item: dict[str, Any] = {
            "class_id": cid,
            "class_name": _class_display_name_or_id(class_names, cid),
            "bbox_xyxy": [float(xyxy[j, k]) for k in range(4)],
        }
        if scores is not None and j < len(scores):
            item["confidence"] = float(scores[j])
        rows.append(item)
    return json.dumps(rows, ensure_ascii=False)


def write_row_dicts_to_csv_path_skip_if_empty(
    csv_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    将「字典行」列表转为 ``DataFrame`` 并写入 ``csv_path``（无行索引列）。

    若 ``rows`` 为空则什么也不做（不创建文件）。写入成功后打印保存路径。
    不创建父目录；调用方需保证 ``csv_path.parent`` 已存在。
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV 已保存: {csv_path}")
