#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 FiftyOne 里已有的头框打分，可选把结果写回这些真值框的 tag。

``--sample-tags`` 选图。眼睛真值读框上的 ``eye_open_*``，多个时取最大编号，
没有则视为 ``eye_open_0``。不改 ``eye_open_*`` 和 ``changed``。默认只打分；
``--write-fo`` 才写回原框。判错写 ``eval_err`` 以及 ``pred_eye_open_<n>``
或 ``pred_background``；判对只写预测标签，不写 ``eval_ok``。再次写入会清掉旧的
``eval_ok``。``--care-classes`` 指定关注类时，ERR 按「是否属于 care」
的 XOR 判定（如只关心 ``eye_open_2``）；未指定则在训练组空间一一对应。

示例::

    export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017   # 可选，本机已有 mongod 时脚本会默认连上
    uv run scripts/val_classifier_with_fiftyone.py \\
        --ckpt logs/.../epoch=19-val_acc=0.96.ckpt \\
        --config configs/experiments/image_classifiter_from_detect_box_head_eye_open_v100.yaml \\
        --dataset BBM08S_head \\
        --sample-tags test_v301 \\
        --box-labels head \\
        --care-classes eye_open_2 \\
        --write-fo
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from lightning.pytorch.cli import instantiate_class
from torch.utils.data import DataLoader
from tqdm import tqdm

from lovely_deep_learning.dataset.base import BaseDataset
from lovely_deep_learning.dataset.box_crop_distribution import box_keep_masks
from lovely_deep_learning.dataset.image_classifier import (
    ImageClassifierDataset,
    ImageClassifierFromDetectBoxDataset,
)

ROOT = Path(__file__).resolve().parents[1]
EYE_TAG_PREFIX = "eye_open_"
EVAL_OK = "eval_ok"
EVAL_ERR = "eval_err"
PRED_TAG_PREFIX = "pred_eye_open_"
PRED_BACKGROUND = "pred_background"
DEFAULT_EXCLUDE_TAGS = ("dup_repeat_drop", "dup_near_drop")


@dataclass
class BoxPrediction:
    """One GT-box classification result."""

    sample_id: str
    detection_id: str
    img_path: Path
    cx: float
    cy: float
    w: float
    h: float
    gt_original_id: int
    gt_original_name: str
    gt_group_id: int
    pred_group_id: int
    pred_group_name: str
    pred_tag: str
    pred_conf: float
    is_err: bool


@dataclass
class EvalStats:
    n_boxes: int = 0
    n_err: int = 0
    n_ok: int = 0
    by_gt: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "err": 0}))

    def add(self, row: BoxPrediction) -> None:
        self.n_boxes += 1
        if row.is_err:
            self.n_err += 1
        else:
            self.n_ok += 1
        bucket = self.by_gt[row.gt_original_name]
        bucket["n"] += 1
        if row.is_err:
            bucket["err"] += 1


class BoxCropEvalDataset(ImageClassifierFromDetectBoxDataset):
    """框级数据集；在 ``class_groups`` 改写前保留原始 ``class_id`` / ``class_name``。"""

    def _filter_boxes(self) -> None:
        table = self.sample_path_table
        groups = self.class_groups if self._has_label else None
        masks = box_keep_masks(
            table,
            min_side_px=self.min_side_px,
            has_label=self._has_label,
            class_groups=groups,
        )
        keep = masks["keep"]
        geo_kept = table.loc[keep].reset_index(drop=True)
        if self._has_label and len(geo_kept):
            self._original_class_ids = [
                int(str(value).strip()) for value in geo_kept["class_id"].tolist()
            ]
            self._original_class_names = [str(name) for name in geo_kept["class_name"].tolist()]
        else:
            self._original_class_ids = []
            self._original_class_names = []
        super()._filter_boxes()
        if self._has_label and len(self._original_class_ids) != len(self.sample_path_table):
            raise RuntimeError(
                "原始标签与筛选后行数不一致："
                f"{len(self._original_class_ids)} vs {len(self.sample_path_table)}"
            )


def nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def comma_names(value: str) -> list[str]:
    names = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one name")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("duplicate names are not allowed")
    return names


def comma_names_optional(value: str) -> list[str]:
    return [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]


def eye_open_code(tags: object) -> int:
    """框上 ``eye_open_<n>`` 的最大 n。没有这种标签时返回 0。"""
    codes: list[int] = []
    for tag in tags or []:
        if not isinstance(tag, str) or not tag.startswith(EYE_TAG_PREFIX):
            continue
        suffix = tag[len(EYE_TAG_PREFIX) :]
        if suffix.isdigit():
            codes.append(int(suffix))
    if not codes:
        return 0
    return max(codes)


def eye_class_name(label: str, code: int) -> str:
    return f"{label}-eye_open_{code}"


def fo_xywh_to_yolo_cxcywh(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x + width / 2.0, y + height / 2.0, width, height


def is_eval_tag(tag: object) -> bool:
    """评估写在真值框上的标签。不含 ``eye_open_*`` 和 ``changed``。"""
    if not isinstance(tag, str):
        return False
    if tag in {EVAL_OK, EVAL_ERR, PRED_BACKGROUND}:
        return True
    if tag.startswith(PRED_TAG_PREFIX) and tag[len(PRED_TAG_PREFIX) :].isdigit():
        return True
    return False


def pred_tag_for_group(
    pred_group_id: int,
    groups: dict[int, list[int]] | None,
    background_id: int | None,
) -> str:
    """训练组 → ``pred_background`` 或 ``pred_eye_open_<最小原始 id>``。"""
    group_id = int(pred_group_id)
    if background_id is not None and group_id == int(background_id):
        return PRED_BACKGROUND
    members = (groups or {}).get(group_id)
    if not members:
        raise ValueError(f"预测组 {group_id} 不在 class_groups 里")
    return f"{PRED_TAG_PREFIX}{min(int(item) for item in members)}"


def strip_eval_tags(tags: object) -> list[str]:
    return [str(tag) for tag in (tags or []) if not is_eval_tag(tag)]


def apply_eval_tags(tags: object, *, is_err: bool, pred_tag: str) -> list[str]:
    """判错写 ``eval_err`` 和预测标签；判对只写预测标签。"""
    updated = strip_eval_tags(tags)
    if is_err:
        updated.append(EVAL_ERR)
    updated.append(pred_tag)
    return updated


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def data_init_args(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data") or {}
    init = data.get("init_args") or {}
    if hasattr(init, "as_dict"):
        init = init.as_dict()
    return dict(init)


def parse_class_groups_from_cfg(init_args: dict[str, Any]) -> tuple[dict[int, list[int]] | None, int | None]:
    raw = init_args.get("class_groups")
    return ImageClassifierDataset.parse_class_groups(raw)


def original_to_group_map(groups: dict[int, list[int]] | None) -> dict[int, int]:
    if not groups:
        return {}
    mapping: dict[int, int] = {}
    for group_id, members in groups.items():
        for original_id in members:
            mapping[int(original_id)] = int(group_id)
    return mapping


def resolve_care_classes(
    care_text: str | None,
    id_to_name: dict[int, str],
) -> set[int] | None:
    """Parse ``--care-classes`` into original class ids.

    Accepts comma-separated original ids (``2``), full names (``head-eye_open_2``),
    or attribute suffixes (``eye_open_2``). ``None`` / empty → exact group match mode.
    """
    if care_text is None:
        return None
    parts = [part.strip() for part in care_text.split(",") if part.strip()]
    if not parts:
        return None
    name_to_ids: dict[str, list[int]] = defaultdict(list)
    for class_id, name in id_to_name.items():
        name_to_ids[name].append(class_id)
        if "-" in name:
            _, _, suffix = name.partition("-")
            if suffix:
                name_to_ids[suffix].append(class_id)
    care: set[int] = set()
    for part in parts:
        if part.lstrip("-").isdigit():
            care.add(int(part))
            continue
        matches = name_to_ids.get(part, [])
        if not matches:
            known = ", ".join(f"{i}:{n}" for i, n in sorted(id_to_name.items()))
            raise ValueError(f"unknown --care-classes entry {part!r}; known: {known}")
        care.update(matches)
    unknown = sorted(care - set(id_to_name))
    if unknown:
        raise ValueError(f"--care-classes ids not in CSV vocabulary: {unknown}")
    return care


def pred_originals_for_group(
    pred_group_id: int,
    groups: dict[int, list[int]] | None,
    background_id: int | None,
) -> set[int]:
    """Original class ids represented by a predicted training group id."""
    if background_id is not None and pred_group_id == background_id:
        return set()
    if not groups:
        return {int(pred_group_id)}
    members = groups.get(int(pred_group_id))
    if members is None:
        return set()
    return {int(item) for item in members}


def is_error(
    *,
    gt_original_id: int,
    pred_group_id: int,
    care_ids: set[int] | None,
    groups: dict[int, list[int]] | None,
    background_id: int | None,
    original_to_group: dict[int, int],
) -> bool:
    """ERR rule: care XOR membership, or exact training-group match when care is unset."""
    if care_ids is not None:
        gt_in_care = gt_original_id in care_ids
        pred_in_care = bool(pred_originals_for_group(pred_group_id, groups, background_id) & care_ids)
        return gt_in_care != pred_in_care
    if groups is None:
        return int(gt_original_id) != int(pred_group_id)
    gt_group = original_to_group.get(int(gt_original_id))
    if gt_group is None:
        return True
    return int(gt_group) != int(pred_group_id)


def strip_pretrained_weight(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Avoid re-downloading official weights when loading a finetuned ckpt."""
    cfg = copy.deepcopy(model_cfg)
    init_args = cfg.get("init_args") or {}
    dag = init_args.get("model")
    if isinstance(dag, dict):
        dag = dict(dag)
        dag["pretrained"] = False
        dag.pop("weight", None)
        init_args = dict(init_args)
        init_args["model"] = dag
        cfg["init_args"] = init_args
    return cfg


def coerce_config_value(value: Any) -> Any:
    """Resolve YAML leftovers such as ``torch.float32`` that LightningCLI normally parses."""
    if isinstance(value, str) and value.startswith("torch."):
        attr = value[len("torch.") :]
        if hasattr(torch, attr):
            return getattr(torch, attr)
    return value


def instantiate_nested(spec: Any) -> Any:
    """Recursively instantiate Lightning-style ``{class_path, init_args}`` trees."""
    if isinstance(spec, dict) and "class_path" in spec:
        raw_init = spec.get("init_args") or {}
        if hasattr(raw_init, "as_dict"):
            raw_init = raw_init.as_dict()
        init_args = {
            key: instantiate_nested(value) for key, value in dict(raw_init).items()
        }
        return instantiate_class((), {"class_path": spec["class_path"], "init_args": init_args})
    if isinstance(spec, list):
        return [instantiate_nested(item) for item in spec]
    if isinstance(spec, tuple):
        return tuple(instantiate_nested(item) for item in spec)
    return coerce_config_value(spec)


def load_module(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict) or "class_path" not in model_cfg:
        raise ValueError(f"config missing model.class_path: {config_path}")
    # BaseModule keeps optimizer / lr_scheduler / DAGNet ``model`` as dicts;
    # only criterion / postprocess / metrics are objects.
    prepared = strip_pretrained_weight(model_cfg)
    init_args = dict(prepared.get("init_args") or {})
    for key in ("criterion", "postprocess", "metrics"):
        if key in init_args:
            init_args[key] = instantiate_nested(init_args[key])
    prepared["init_args"] = init_args
    module = instantiate_class((), prepared)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    try:
        incompatible = module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint does not match --config (check class_groups / num_classes / head size). "
            f"detail: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"warning: missing={list(incompatible.missing_keys)[:5]} "
            f"unexpected={list(incompatible.unexpected_keys)[:5]}",
            file=sys.stderr,
        )
    module.eval()
    module.to(device)
    return module, cfg


def build_dataset(cfg: dict[str, Any], csv_path: Path) -> BoxCropEvalDataset:
    init_args = data_init_args(cfg)
    transform_spec = init_args.get("transform_val")
    if transform_spec is None:
        raise ValueError("config data.init_args.transform_val is required")
    transform = instantiate_nested(transform_spec)
    key_map = init_args.get("key_map")
    return BoxCropEvalDataset(
        csv_paths=[csv_path],
        key_map=key_map,
        transform=transform,
        norm_mean=init_args.get("norm_mean"),
        norm_std=init_args.get("norm_std"),
        box_scale=float(init_args.get("box_scale", 1.2)),
        min_side_px=float(init_args.get("min_side_px", 16.0)),
        class_groups=init_args.get("class_groups"),
        background_ratio=0.0,
        background_iou=float(init_args.get("background_iou", 0.3)),
    )


def append_problem(problems: list[dict], **fields: object) -> None:
    problems.append({key: value for key, value in fields.items()})


def image_size(sample) -> tuple[int, int] | None:
    meta = getattr(sample, "metadata", None)
    width = getattr(meta, "width", None) if meta is not None else None
    height = getattr(meta, "height", None) if meta is not None else None
    if width and height:
        return int(width), int(height)
    path = Path(str(sample.filepath))
    if not path.is_file():
        return None
    from PIL import Image

    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def collect_labeled_boxes(
    dataset,
    *,
    sample_tags: list[str],
    exclude_tags: list[str],
    label_field: str,
    box_labels: set[str],
    problems: list[dict],
) -> tuple[list[dict[str, object]], list[str]]:
    """从 FiftyOne 框收集分类 CSV 行，并返回本次选中的样本 id。"""
    import fiftyone as fo

    if label_field not in dataset.get_field_schema():
        raise ValueError(f"数据集没有字段 {label_field}")
    field = dataset.get_field(label_field)
    if not issubclass(field.document_type, fo.Detections):
        raise ValueError(f"字段 {label_field} 不是 Detections")
    view = dataset.match_tags(sample_tags, bool=True, all=True)
    excluded = set(exclude_tags)
    rows: list[dict[str, object]] = []
    sample_ids: list[str] = []
    for sample in view.iter_samples(progress=True, autosave=False):
        tags = {str(tag) for tag in (sample.tags or [])}
        if tags & excluded:
            continue
        size = image_size(sample)
        if size is None:
            append_problem(
                problems,
                sample_id=str(sample.id),
                file=str(sample.filepath),
                reason="missing_image",
            )
            continue
        sample_ids.append(str(sample.id))
        img_w, img_h = size
        label = sample[label_field]
        for detection in list(getattr(label, "detections", None) or []):
            if str(detection.label) not in box_labels:
                continue
            box = list(getattr(detection, "bounding_box", None) or [])
            if len(box) != 4:
                append_problem(
                    problems,
                    sample_id=str(sample.id),
                    detection_id=str(getattr(detection, "id", "")),
                    reason="bad_box",
                )
                continue
            cx, cy, width, height = fo_xywh_to_yolo_cxcywh(box)
            if width <= 0 or height <= 0:
                append_problem(
                    problems,
                    sample_id=str(sample.id),
                    detection_id=str(detection.id),
                    reason="invalid_box",
                )
                continue
            code = eye_open_code(detection.tags)
            det_label = str(detection.label)
            rows.append(
                {
                    "path_img": str(sample.filepath),
                    "class_name": eye_class_name(det_label, code),
                    "class_id": code,
                    "cx": cx,
                    "cy": cy,
                    "w": width,
                    "h": height,
                    "img_w": img_w,
                    "img_h": img_h,
                    "sample_id": str(sample.id),
                    "detection_id": str(detection.id),
                }
            )
    return rows, sample_ids


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_inference(
    module,
    dataset: BoxCropEvalDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    care_ids: set[int] | None,
) -> list[BoxPrediction]:
    groups = dataset.class_groups
    background_id = dataset.background_id
    original_to_group = original_to_group_map(groups)
    collate = ImageClassifierDataset.get_collate_fn_for_dataloader()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    rows: list[BoxPrediction] = []
    offset = 0
    with torch.inference_mode():
        for net_in, net_out in tqdm(loader, desc="infer", unit="batch"):
            imgs = BaseDataset.stack_batch_images(net_in)
            imgs = imgs.to(device, non_blocking=True)
            preds = module(imgs)
            metric_preds = module.postprocess.run(preds)
            pred_ids = metric_preds["pred_ids"].detach().cpu()
            pred_conf = metric_preds["pred_conf"].detach().cpu()
            batch_n = len(net_in)
            for i in range(batch_n):
                index = offset + i
                img_path = Path(str(net_in[i]["img_path"])).expanduser().resolve()
                table_row = dataset.sample_path_table.iloc[index]
                gt_original_id = int(dataset._original_class_ids[index])
                gt_original_name = str(dataset._original_class_names[index])
                gt_group_id = int(net_out["class_id"][i].item()) if net_out else int(
                    original_to_group.get(gt_original_id, -1)
                )
                pred_group_id = int(pred_ids[i].item())
                pred_group_name = dataset.class_name_for_id(pred_group_id)
                pred_tag = pred_tag_for_group(pred_group_id, groups, background_id)
                conf = float(pred_conf[i].item())
                err = is_error(
                    gt_original_id=gt_original_id,
                    pred_group_id=pred_group_id,
                    care_ids=care_ids,
                    groups=groups,
                    background_id=background_id,
                    original_to_group=original_to_group,
                )
                rows.append(
                    BoxPrediction(
                        sample_id=str(table_row["sample_id"]),
                        detection_id=str(table_row["detection_id"]),
                        img_path=img_path,
                        cx=float(table_row["cx"]),
                        cy=float(table_row["cy"]),
                        w=float(table_row["w"]),
                        h=float(table_row["h"]),
                        gt_original_id=gt_original_id,
                        gt_original_name=gt_original_name,
                        gt_group_id=gt_group_id,
                        pred_group_id=pred_group_id,
                        pred_group_name=pred_group_name,
                        pred_tag=pred_tag,
                        pred_conf=conf,
                        is_err=err,
                    )
                )
            offset += batch_n
    return rows


def ensure_fiftyone_database_uri() -> None:
    """Reuse an already-running mongod (common on this machine) instead of spawning one."""
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")


def write_fiftyone(
    *,
    dataset,
    rows: list[BoxPrediction],
    sample_ids: list[str],
    label_field: str,
    box_labels: set[str],
    problems: list[dict],
) -> int:
    """清掉选中框上的旧评估标签，再给打分框写上本次结果。"""
    by_detection = {row.detection_id: row for row in rows}
    tagged = 0
    for sample_id in tqdm(sample_ids, desc="write-fo", unit="sample"):
        sample = dataset[sample_id]
        label = sample[label_field]
        detections = list(getattr(label, "detections", None) or [])
        changed = False
        seen: set[str] = set()
        for detection in detections:
            if str(detection.label) not in box_labels:
                continue
            detection_id = str(detection.id)
            seen.add(detection_id)
            result = by_detection.get(detection_id)
            if result is None:
                new_tags = strip_eval_tags(detection.tags)
            else:
                new_tags = apply_eval_tags(detection.tags, is_err=result.is_err, pred_tag=result.pred_tag)
                tagged += 1
            if [str(tag) for tag in (detection.tags or [])] != new_tags:
                detection.tags = new_tags
                changed = True
        if changed:
            sample[label_field] = label
            sample.save()
        for row in rows:
            if row.sample_id == sample_id and row.detection_id not in seen:
                append_problem(
                    problems,
                    sample_id=sample_id,
                    detection_id=row.detection_id,
                    reason="missing_detection",
                )
    return tagged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, type=Path, help="Lightning .ckpt path")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML (must match ckpt head)")
    parser.add_argument("--dataset", required=True, type=nonempty, help="FiftyOne dataset name")
    parser.add_argument("--sample-tags", required=True, type=comma_names, help="样本须同时带有这些 tag")
    parser.add_argument("--box-labels", required=True, type=comma_names, help="要评估的 detection.label，例如 head")
    parser.add_argument("--label-field", default="ground_truth", help="检测框字段，默认 ground_truth")
    parser.add_argument(
        "--exclude-tags",
        type=comma_names_optional,
        default=list(DEFAULT_EXCLUDE_TAGS),
        help="带其中任一 tag 的样本跳过。默认 dup_repeat_drop,dup_near_drop",
    )
    parser.add_argument(
        "--care-classes",
        default=None,
        help="Comma-separated focus classes (id / name / eye_open_2). Omit for exact group match.",
    )
    parser.add_argument("--write-fo", action="store_true", help="判错写 eval_err 和预测标签；判对只写预测标签")
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0 (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Save summary/problems (default: logs/val_classifier_fo/<stamp>)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = args.ckpt.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"error: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module, cfg = load_module(config_path, ckpt_path, device)
    init_args = data_init_args(cfg)
    groups, background_id = parse_class_groups_from_cfg(init_args)

    ensure_fiftyone_database_uri()
    try:
        import fiftyone as fo
    except ImportError:
        print("error: fiftyone is not installed in this environment", file=sys.stderr)
        return 2
    if not fo.dataset_exists(args.dataset):
        print(f"error: FiftyOne dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    fo_dataset = fo.load_dataset(args.dataset)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out_dir or (ROOT / "logs" / "val_classifier_fo" / stamp)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[dict] = []
    try:
        labeled_rows, sample_ids = collect_labeled_boxes(
            fo_dataset,
            sample_tags=args.sample_tags,
            exclude_tags=args.exclude_tags,
            label_field=args.label_field,
            box_labels=set(args.box_labels),
            problems=problems,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not labeled_rows:
        print("error: 没有样本同时带有 --sample-tags，或这些样本没有可评估的框", file=sys.stderr)
        return 2
    csv_path = out_dir / "boxes.csv"
    pd.DataFrame(labeled_rows).to_csv(csv_path, index=False)
    try:
        dataset = build_dataset(cfg, csv_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    id_to_name = ImageClassifierDataset.names_by_original_id(
        pd.DataFrame(
            {
                "class_id": dataset._original_class_ids,
                "class_name": dataset._original_class_names,
            }
        )
    )
    try:
        care_ids = resolve_care_classes(args.care_classes, id_to_name)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    batch_size = int(args.batch_size or init_args.get("batch_size") or 32)
    num_workers = int(args.num_workers if args.num_workers is not None else init_args.get("num_workers") or 0)

    print(f"device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"config={config_path}")
    print(f"dataset={args.dataset} label_field={args.label_field} boxes={len(dataset)}")
    print(f"sample_tags={args.sample_tags} exclude_tags={args.exclude_tags} box_labels={args.box_labels}")
    print(f"class_groups={groups} background_id={background_id}")
    print(f"care_classes={sorted(care_ids) if care_ids is not None else None}")
    print(f"write_fo={args.write_fo}")

    rows = run_inference(
        module,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        care_ids=care_ids,
    )
    stats = EvalStats()
    for row in rows:
        stats.add(row)

    results_csv = out_dir / "box_results.csv"
    pd.DataFrame(
        [
            {
                "sample_id": row.sample_id,
                "detection_id": row.detection_id,
                "img_path": str(row.img_path),
                "cx": row.cx,
                "cy": row.cy,
                "w": row.w,
                "h": row.h,
                "gt_original_id": row.gt_original_id,
                "gt_original_name": row.gt_original_name,
                "gt_group_id": row.gt_group_id,
                "pred_group_id": row.pred_group_id,
                "pred_group_name": row.pred_group_name,
                "pred_tag": row.pred_tag,
                "pred_conf": row.pred_conf,
                "is_err": row.is_err,
            }
            for row in rows
        ]
    ).to_csv(results_csv, index=False)

    n_tagged = 0
    if args.write_fo:
        n_tagged = write_fiftyone(
            dataset=fo_dataset,
            rows=rows,
            sample_ids=sample_ids,
            label_field=args.label_field,
            box_labels=set(args.box_labels),
            problems=problems,
        )

    problems_path = out_dir / "problems.jsonl"
    with problems_path.open("w", encoding="utf-8") as handle:
        for item in problems:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "ckpt": str(ckpt_path),
        "config": str(config_path),
        "dataset": args.dataset,
        "sample_tags": args.sample_tags,
        "exclude_tags": args.exclude_tags,
        "box_labels": args.box_labels,
        "label_field": args.label_field,
        "boxes_csv": str(csv_path),
        "write_fo": args.write_fo,
        "eval_tags": [EVAL_ERR],
        "care_classes": sorted(care_ids) if care_ids is not None else None,
        "care_class_names": (
            [id_to_name[i] for i in sorted(care_ids) if i in id_to_name] if care_ids is not None else None
        ),
        "class_groups": {str(k): v for k, v in (groups or {}).items()},
        "background_id": background_id,
        "device": str(device),
        "n_boxes": stats.n_boxes,
        "n_ok": stats.n_ok,
        "n_err": stats.n_err,
        "err_rate": (stats.n_err / stats.n_boxes) if stats.n_boxes else 0.0,
        "by_gt": dict(stats.by_gt),
        "n_tagged": n_tagged,
        "n_problems": len(problems),
        "results_csv": str(results_csv),
        "problems_path": str(problems_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)

    print(
        f"boxes={stats.n_boxes} ok={stats.n_ok} err={stats.n_err} "
        f"err_rate={summary['err_rate']:.4f}"
    )
    for name, bucket in sorted(stats.by_gt.items()):
        rate = (bucket["err"] / bucket["n"]) if bucket["n"] else 0.0
        print(f"  gt={name} n={bucket['n']} err={bucket['err']} err_rate={rate:.4f}")
    if args.write_fo:
        print(f"tagged={n_tagged}")
        print(f"app filter: detection tag `{EVAL_ERR}`")
    print(f"results_csv={results_csv}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"problems_path={problems_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
