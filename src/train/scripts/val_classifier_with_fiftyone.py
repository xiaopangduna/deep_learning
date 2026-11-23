#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 FiftyOne 里已有的框打分，可选把结果写回这些框的 tag。

``--sample-tags`` 选图。``--class-map`` 给出模型输出 id 到框上的 tag，id 和 tag
一一对应，例如 ``0:face_frontal,1:face_side,2:face_head``。值必须和框上的 tag
完全一致。``--box-labels head`` 
只认映射里的名字，``changed`` 和旧的评估 tag 不参与。

在带这些样本 tag 的图上，``--box-labels`` 里的每个框都会看一遍。短边小于配置
``min_side_px`` 的框跳过，不打分。正好命中一个映射 tag 的框裁图打分：判错写
``eval_err`` 和 ``pred_<tag>``，判对只写预测标签。一个映射 tag 都没有的框也裁图
推理，写 ``eval_bad_tag`` 和 ``pred_<tag>``，不进对错统计，预测可当预标注。命中
多个映射 tag 时只写 ``eval_bad_tag``，不进模型。预测 id 不在映射里时，``pred_``
后面是这个 id。
不改真值 tag 和 ``changed``。默认只打分；``--write-fo`` 才写回原框。再次写入会清掉旧的
``eval_ok``、``eval_err``、``eval_bad_tag`` 和 ``pred_*``。

示例::

    export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017   # 可选，本机已有 mongod 时脚本会默认连上
    uv run scripts/val_classifier_with_fiftyone.py 
        --ckpt logs/.../epoch=19-val_acc=0.96.ckpt 
        --config configs/experiments/image_classifiter_from_detect_box_head_face_mobilenet_v2_v001.yaml 
        --dataset BBM08S_head 
        --sample-tags test_v401 
        --box-labels head 
        --class-map 0:face_frontal,1:face_side,2:face_head 
        --write-fo
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import OrderedDict, defaultdict
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
EVAL_OK = "eval_ok"
EVAL_ERR = "eval_err"
EVAL_BAD = "eval_bad_tag"
PRED_TAG_PREFIX = "pred_"
STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_ERR = "err"
STATUS_BAD = "bad_tag"
REASON_NO_GT = "no_gt_tag"
REASON_MULTI_GT = "multi_gt_tag"
FO_SAMPLE_BATCH = 100
IMG_CACHE_SIZE = 16


@dataclass
class ClassMap:
    """模型输出 id 与框 tag 一一对应。"""

    entries: list[tuple[int, str]]
    name_to_id: dict[str, int]
    id_to_names: dict[int, list[str]]

    def first_name(self, model_id: int) -> str | None:
        names = self.id_to_names.get(int(model_id))
        if not names:
            return None
        return names[0]

    def ids(self) -> set[int]:
        return set(self.id_to_names)


@dataclass
class BoxVerdict:
    """一个框的真值匹配结果；打分后补上预测。"""

    sample_id: str
    detection_id: str
    img_path: str
    img_w: int
    img_h: int
    cx: float
    cy: float
    w: float
    h: float
    status: str
    gt_name: str = ""
    gt_id: int = -1
    pred_group_id: int = -1
    pred_group_name: str = ""
    pred_tag: str = ""
    pred_conf: float = float("nan")
    bad_reason: str = ""
    matched_names: tuple[str, ...] = ()


@dataclass
class EvalStats:
    n_scored: int = 0
    n_err: int = 0
    n_ok: int = 0
    n_bad: int = 0
    n_no_gt: int = 0
    n_multi_gt: int = 0
    by_gt: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "err": 0}))

    def add(self, row: BoxVerdict) -> None:
        if row.status == STATUS_BAD:
            self.n_bad += 1
            if row.bad_reason == REASON_NO_GT:
                self.n_no_gt += 1
            elif row.bad_reason == REASON_MULTI_GT:
                self.n_multi_gt += 1
            return
        self.n_scored += 1
        if row.status == STATUS_ERR:
            self.n_err += 1
        else:
            self.n_ok += 1
        bucket = self.by_gt[row.gt_name]
        bucket["n"] += 1
        if row.status == STATUS_ERR:
            bucket["err"] += 1


class BoxCropEvalDataset(ImageClassifierFromDetectBoxDataset):
    """框级数据集；在 ``class_groups`` 改写前保留原始 ``class_id`` / ``class_name``。"""

    def _read_img_cached(self, img_path: str):
        """同一 worker 里按路径缓存整图，连续同图多框只解码一次。"""
        cache = getattr(self, "_img_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._img_cache = cache
        hit = cache.get(img_path)
        if hit is not None:
            cache.move_to_end(img_path)
            return hit
        decoded = self.read_img(img_path, None)
        cache[img_path] = decoded
        while len(cache) > IMG_CACHE_SIZE:
            cache.popitem(last=False)
        return decoded

    def __getitem__(self, index):
        n_positive = len(self.sample_path_table)
        if index >= n_positive:
            return self._background_item()
        row = self.sample_path_table.iloc[index]
        img_path = str(row["img_path"])
        img_np, img_shape = self._read_img_cached(img_path)
        crop = self.crop_cxcywh(
            img_np,
            float(row["cx"]),
            float(row["cy"]),
            float(row["w"]),
            float(row["h"]),
            self.box_scale,
        )
        class_name = row["class_name"] if self._has_label else None
        class_id = int(row["class_id"]) if self._has_label else None
        return self._pack_crop(img_path, img_shape, crop, class_name, class_id)

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
        # 评估子集可以缺某个训练类，不要求 class_groups 里每一组都有样本。
        require_samples = ImageClassifierFromDetectBoxDataset.require_merged_samples
        ImageClassifierFromDetectBoxDataset.require_merged_samples = staticmethod(
            lambda table, groups: None
        )
        try:
            super()._filter_boxes()
        finally:
            ImageClassifierFromDetectBoxDataset.require_merged_samples = staticmethod(require_samples)
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


def parse_class_map(value: str) -> ClassMap:
    """解析 ``0:face_frontal,1:face_side``。id 和 tag 都不能重复。"""
    parts = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("--class-map 不能为空")
    entries: list[tuple[int, str]] = []
    name_to_id: dict[str, int] = {}
    id_to_names: dict[int, list[str]] = defaultdict(list)
    for part in parts:
        id_text, sep, name = part.partition(":")
        id_text = id_text.strip()
        name = name.strip()
        if not sep or not id_text or not name:
            raise ValueError(f"--class-map 项须为 id:类名，收到 {part!r}")
        if not id_text.isdigit():
            raise ValueError(f"--class-map 的 id 须为非负整数，收到 {id_text!r}")
        model_id = int(id_text)
        if model_id in id_to_names:
            raise ValueError(f"--class-map 的 id 重复: {model_id}")
        if name in name_to_id:
            raise ValueError(f"--class-map 类名重复: {name}")
        entries.append((model_id, name))
        name_to_id[name] = model_id
        id_to_names[model_id].append(name)
    return ClassMap(entries=entries, name_to_id=name_to_id, id_to_names=dict(id_to_names))


def class_map_arg(value: str) -> ClassMap:
    try:
        return parse_class_map(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_class_map(
    class_map: ClassMap,
    groups: dict[int, list[int]] | None,
    background_id: int | None,
    box_labels: list[str],
) -> None:
    """映射 id 必须是模型输出的前景类，名字必须是框上的 tag，不带框标签前缀。"""
    if background_id is not None and int(background_id) in class_map.ids():
        raise ValueError(
            f"--class-map 不能把 background id {int(background_id)} 当作真值类"
        )
    if groups is not None:
        unknown = sorted(class_map.ids() - set(groups))
        if unknown:
            raise ValueError(
                f"--class-map 的 id 不在 class_groups 里: {unknown}；"
                f"模型输出组为 {sorted(groups)}"
            )
    for box_label in box_labels:
        prefix = f"{box_label}-"
        prefixed = [name for name in class_map.name_to_id if name.startswith(prefix)]
        if prefixed:
            raise ValueError(
                f"--class-map 写框上的 tag，不要带框标签前缀 {prefix!r}: {prefixed}"
            )


def eval_class_groups(
    groups: dict[int, list[int]] | None,
    background_id: int | None,
) -> dict[int, list[int] | str] | None:
    """评估表的 class_id 已经是模型输出 id，分组改成恒等映射。"""
    if not groups and background_id is None:
        return None
    synthetic: dict[int, list[int] | str] = {}
    for model_id in groups or {}:
        synthetic[int(model_id)] = [int(model_id)]
    if background_id is not None:
        synthetic[int(background_id)] = ImageClassifierDataset.BACKGROUND_NAME
    return synthetic


def matched_class_names(box_label: str, tags: object, class_map: ClassMap) -> list[str]:
    """按映射顺序返回与框 tag 完全相同的类名。``box_label`` 不参与匹配。"""
    del box_label
    present = {str(tag) for tag in (tags or []) if isinstance(tag, str)}
    return [name for name in class_map.name_to_id if name in present]


def fo_xywh_to_yolo_cxcywh(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x + width / 2.0, y + height / 2.0, width, height


def is_eval_tag(tag: object) -> bool:
    """评估写在框上的标签。真值类名和 ``changed`` 保留。"""
    if not isinstance(tag, str):
        return False
    if tag in {EVAL_OK, EVAL_ERR, EVAL_BAD}:
        return True
    return tag.startswith(PRED_TAG_PREFIX)


def pred_tag_for_id(pred_id: int, class_map: ClassMap) -> str:
    """映射内的模型 id → ``pred_<tag>``；映射外的 id → ``pred_<id>``。"""
    group_id = int(pred_id)
    name = class_map.first_name(group_id)
    if name is not None:
        return f"{PRED_TAG_PREFIX}{name}"
    return f"{PRED_TAG_PREFIX}{group_id}"


def strip_eval_tags(tags: object) -> list[str]:
    return [str(tag) for tag in (tags or []) if not is_eval_tag(tag)]


def apply_eval_tags(tags: object, *, status: str, pred_tag: str) -> list[str]:
    """判错写 ``eval_err`` 和预测标签；判对只写预测标签；标注异常写 ``eval_bad_tag``，有预测时一并写上。"""
    updated = strip_eval_tags(tags)
    if status == STATUS_BAD:
        updated.append(EVAL_BAD)
        if pred_tag:
            updated.append(pred_tag)
        return updated
    if status == STATUS_ERR:
        updated.append(EVAL_ERR)
    if pred_tag:
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


def build_dataset(
    cfg: dict[str, Any],
    csv_path: Path,
    class_groups: dict[int, list[int] | str] | None = None,
) -> BoxCropEvalDataset:
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
        class_groups=init_args.get("class_groups") if class_groups is None else class_groups,
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


def collect_eval_view(
    dataset,
    *,
    sample_tags: list[str],
    label_field: str,
):
    """带 tag 过滤和字段裁剪的评估视图；缺 metadata 时补一次。"""
    import fiftyone as fo

    if label_field not in dataset.get_field_schema():
        raise ValueError(f"数据集没有字段 {label_field}")
    field = dataset.get_field(label_field)
    if not issubclass(field.document_type, fo.Detections):
        raise ValueError(f"字段 {label_field} 不是 Detections")
    view = dataset.match_tags(sample_tags, bool=True, all=True)
    schema = dataset.get_field_schema()
    keep = [name for name in ("filepath", "tags", "metadata", label_field) if name in schema]
    if keep:
        view = view.select_fields(keep)
    if "metadata" in schema and view.exists("metadata.width", False).limit(1).count() > 0:
        view.compute_metadata(overwrite=False)
    return view


def collect_labeled_boxes(
    dataset,
    *,
    sample_tags: list[str],
    label_field: str,
    box_labels: set[str],
    class_map: ClassMap,
    min_side_px: float,
    problems: list[dict],
) -> tuple[list[BoxVerdict], list[str]]:
    """收集框判决。正好一个类名的框待打分；没有类名的框也待推理；多个类名标为标注异常。"""
    view = collect_eval_view(
        dataset,
        sample_tags=sample_tags,
        label_field=label_field,
    )
    verdicts: list[BoxVerdict] = []
    sample_ids: list[str] = []
    for sample in view.iter_samples(progress=True, autosave=False, batch_size=FO_SAMPLE_BATCH):
        size = image_size(sample)
        if size is None:
            append_problem(
                problems,
                sample_id=str(sample.id),
                file=str(sample.filepath),
                reason="missing_image",
            )
            continue
        img_w, img_h = size
        label = sample[label_field]
        sample_id = str(sample.id)
        filepath = str(sample.filepath)
        saw_box = False
        for detection in list(getattr(label, "detections", None) or []):
            det_label = str(detection.label)
            if det_label not in box_labels:
                continue
            saw_box = True
            box = list(getattr(detection, "bounding_box", None) or [])
            detection_id = str(getattr(detection, "id", ""))
            if len(box) != 4:
                append_problem(
                    problems,
                    sample_id=sample_id,
                    detection_id=detection_id,
                    reason="bad_box",
                )
                continue
            cx, cy, width, height = fo_xywh_to_yolo_cxcywh(box)
            if width <= 0 or height <= 0:
                append_problem(
                    problems,
                    sample_id=sample_id,
                    detection_id=detection_id,
                    reason="invalid_box",
                )
                continue
            short_side = min(width * img_w, height * img_h)
            if min_side_px > 0 and short_side < min_side_px:
                append_problem(
                    problems,
                    sample_id=sample_id,
                    detection_id=detection_id,
                    reason="short_side",
                )
                continue
            names = matched_class_names(det_label, detection.tags, class_map)
            if len(names) == 1:
                gt_name = names[0]
                verdicts.append(
                    BoxVerdict(
                        sample_id=sample_id,
                        detection_id=detection_id,
                        img_path=filepath,
                        img_w=img_w,
                        img_h=img_h,
                        cx=cx,
                        cy=cy,
                        w=width,
                        h=height,
                        status=STATUS_PENDING,
                        gt_name=gt_name,
                        gt_id=class_map.name_to_id[gt_name],
                        matched_names=(gt_name,),
                    )
                )
                continue
            if not names:
                verdicts.append(
                    BoxVerdict(
                        sample_id=sample_id,
                        detection_id=detection_id,
                        img_path=filepath,
                        img_w=img_w,
                        img_h=img_h,
                        cx=cx,
                        cy=cy,
                        w=width,
                        h=height,
                        status=STATUS_PENDING,
                        gt_id=-1,
                        bad_reason=REASON_NO_GT,
                    )
                )
                append_problem(
                    problems,
                    sample_id=sample_id,
                    detection_id=detection_id,
                    reason=REASON_NO_GT,
                    matched=[],
                )
                continue
            verdicts.append(
                BoxVerdict(
                    sample_id=sample_id,
                    detection_id=detection_id,
                    img_path=filepath,
                    img_w=img_w,
                    img_h=img_h,
                    cx=cx,
                    cy=cy,
                    w=width,
                    h=height,
                    status=STATUS_BAD,
                    bad_reason=REASON_MULTI_GT,
                    matched_names=tuple(names),
                )
            )
            append_problem(
                problems,
                sample_id=sample_id,
                detection_id=detection_id,
                reason=REASON_MULTI_GT,
                matched=list(names),
            )
        if saw_box:
            sample_ids.append(sample_id)
    return verdicts, sample_ids


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_inference(
    module,
    dataset: BoxCropEvalDataset,
    verdicts: dict[str, BoxVerdict],
    *,
    class_map: ClassMap,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> set[str]:
    """给待打分框写入预测，并返回实际打到的 detection id。"""
    collate = ImageClassifierDataset.get_collate_fn_for_dataloader()
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    table = dataset.sample_path_table
    detection_ids = table["detection_id"].astype(str).to_numpy()
    model_ids = [int(str(value).strip()) for value in table["model_id"].tolist()]
    gt_names = table["gt_name"].fillna("").astype(str).tolist()
    seen: set[str] = set()
    offset = 0
    with torch.inference_mode():
        for net_in, _net_out in tqdm(loader, desc="infer", unit="batch"):
            imgs = BaseDataset.stack_batch_images(net_in)
            imgs = imgs.to(device, non_blocking=True)
            preds = module(imgs)
            metric_preds = module.postprocess.run(preds)
            pred_ids = metric_preds["pred_ids"].detach().cpu()
            pred_conf = metric_preds["pred_conf"].detach().cpu()
            batch_n = len(net_in)
            for i in range(batch_n):
                index = offset + i
                detection_id = str(detection_ids[index])
                verdict = verdicts.get(detection_id)
                if verdict is None:
                    continue
                gt_id = int(model_ids[index])
                pred_group_id = int(pred_ids[i].item())
                mapped_name = class_map.first_name(pred_group_id)
                pred_group_name = mapped_name if mapped_name is not None else str(pred_group_id)
                verdict.gt_id = gt_id
                verdict.gt_name = str(gt_names[index])
                verdict.pred_group_id = pred_group_id
                verdict.pred_group_name = pred_group_name
                verdict.pred_tag = pred_tag_for_id(pred_group_id, class_map)
                verdict.pred_conf = float(pred_conf[i].item())
                if gt_id < 0:
                    verdict.status = STATUS_BAD
                    if not verdict.bad_reason:
                        verdict.bad_reason = REASON_NO_GT
                else:
                    verdict.status = STATUS_ERR if pred_group_id != gt_id else STATUS_OK
                    verdict.bad_reason = ""
                seen.add(detection_id)
            offset += batch_n
    return seen


def ensure_fiftyone_database_uri() -> None:
    """Reuse an already-running mongod (common on this machine) instead of spawning one."""
    os.environ.setdefault("FIFTYONE_DATABASE_URI", "mongodb://127.0.0.1:27017")


def write_fiftyone(
    *,
    dataset,
    verdicts: list[BoxVerdict],
    sample_ids: list[str],
    label_field: str,
    box_labels: set[str],
    problems: list[dict],
) -> int:
    """清掉选中框上的旧评估标签，再给打分框写上本次结果。"""
    ids = list(dict.fromkeys(sample_ids))
    if not ids:
        return 0
    by_detection = {
        row.detection_id: row for row in verdicts if row.status in {STATUS_OK, STATUS_ERR, STATUS_BAD}
    }
    by_sample: dict[str, list[BoxVerdict]] = defaultdict(list)
    for row in by_detection.values():
        by_sample[row.sample_id].append(row)
    tagged = 0
    # 不要 select_fields：从裁剪视图 save 可能清掉未选字段。
    view = dataset.select(ids)
    print("write-fo")
    for sample in view.iter_samples(progress=True, autosave=True, batch_size=FO_SAMPLE_BATCH):
        sample_id = str(sample.id)
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
                new_tags = apply_eval_tags(detection.tags, status=result.status, pred_tag=result.pred_tag)
                tagged += 1
            if [str(tag) for tag in (detection.tags or [])] != new_tags:
                detection.tags = new_tags
                changed = True
        if changed:
            sample[label_field] = label
        for row in by_sample.get(sample_id, ()):
            if row.detection_id not in seen:
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
        "--class-map",
        required=True,
        type=class_map_arg,
        help="模型输出 id 到框 tag，一一对应。例如 0:face_frontal,1:face_side,2:face_head",
    )
    parser.add_argument(
        "--write-fo",
        action="store_true",
        help="判错写 eval_err 和预测标签；判对只写预测标签；无真值写 eval_bad_tag 和预测标签；多个真值只写 eval_bad_tag",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0 (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--num-workers", type=int, default=4, help="Override data.num_workers")
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
    class_map: ClassMap = args.class_map
    try:
        validate_class_map(class_map, groups, background_id, args.box_labels)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

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
    min_side_px = float(init_args.get("min_side_px", 16.0))
    try:
        verdicts, sample_ids = collect_labeled_boxes(
            fo_dataset,
            sample_tags=args.sample_tags,
            label_field=args.label_field,
            box_labels=set(args.box_labels),
            class_map=class_map,
            min_side_px=min_side_px,
            problems=problems,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    pending = [row for row in verdicts if row.status == STATUS_PENDING]
    if not verdicts:
        problems_path = out_dir / "problems.jsonl"
        with problems_path.open("w", encoding="utf-8") as handle:
            for item in problems:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("error: 没有样本同时带有 --sample-tags，或这些样本没有可评估的框", file=sys.stderr)
        print(f"problems_path={problems_path}", file=sys.stderr)
        return 2

    batch_size = int(args.batch_size or init_args.get("batch_size") or 32)
    num_workers = int(args.num_workers if args.num_workers is not None else init_args.get("num_workers") or 0)
    csv_path = out_dir / "boxes.csv"
    print(f"device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"config={config_path}")
    print(f"dataset={args.dataset} label_field={args.label_field} pending={len(pending)} bad={len(verdicts) - len(pending)}")
    print(f"sample_tags={args.sample_tags} box_labels={args.box_labels}")
    print(f"class_groups={groups} background_id={background_id}")
    print(f"class_map={class_map.entries}")
    print(f"write_fo={args.write_fo}")

    if pending:
        # 无真值的框没有 class_id。CSV 里借用一个映射内 id 通过类别筛选，model_id 仍为 -1。
        placeholder_id = class_map.entries[0][0]
        pd.DataFrame(
            [
                {
                    "path_img": row.img_path,
                    "class_name": class_map.first_name(
                        row.gt_id if row.gt_id >= 0 else placeholder_id
                    ),
                    "class_id": row.gt_id if row.gt_id >= 0 else placeholder_id,
                    "cx": row.cx,
                    "cy": row.cy,
                    "w": row.w,
                    "h": row.h,
                    "img_w": row.img_w,
                    "img_h": row.img_h,
                    "sample_id": row.sample_id,
                    "detection_id": row.detection_id,
                    "model_id": row.gt_id,
                    "gt_name": row.gt_name,
                }
                for row in pending
            ]
        ).to_csv(csv_path, index=False)
        try:
            dataset = build_dataset(
                cfg,
                csv_path,
                class_groups=eval_class_groups(groups, background_id),
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        by_detection = {row.detection_id: row for row in pending}
        seen = run_inference(
            module,
            dataset,
            by_detection,
            class_map=class_map,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for row in pending:
            if row.detection_id in seen:
                continue
            row.status = STATUS_BAD
            row.bad_reason = "filtered_out"
            append_problem(
                problems,
                sample_id=row.sample_id,
                detection_id=row.detection_id,
                reason="filtered_out",
            )
    else:
        pd.DataFrame(
            columns=[
                "path_img",
                "class_name",
                "class_id",
                "cx",
                "cy",
                "w",
                "h",
                "img_w",
                "img_h",
                "sample_id",
                "detection_id",
                "model_id",
                "gt_name",
            ]
        ).to_csv(csv_path, index=False)

    stats = EvalStats()
    for row in verdicts:
        if row.status == STATUS_PENDING:
            continue
        stats.add(row)

    results_csv = out_dir / "box_results.csv"
    pd.DataFrame(
        [
            {
                "sample_id": row.sample_id,
                "detection_id": row.detection_id,
                "img_path": row.img_path,
                "cx": row.cx,
                "cy": row.cy,
                "w": row.w,
                "h": row.h,
                "status": row.status,
                "gt_name": row.gt_name,
                "gt_id": row.gt_id,
                "matched_names": ",".join(row.matched_names),
                "pred_group_id": row.pred_group_id,
                "pred_group_name": row.pred_group_name,
                "pred_tag": row.pred_tag,
                "pred_conf": row.pred_conf,
                "bad_reason": row.bad_reason,
            }
            for row in verdicts
            if row.status != STATUS_PENDING
        ]
    ).to_csv(results_csv, index=False)

    n_tagged = 0
    if args.write_fo:
        n_tagged = write_fiftyone(
            dataset=fo_dataset,
            verdicts=verdicts,
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
        "box_labels": args.box_labels,
        "label_field": args.label_field,
        "boxes_csv": str(csv_path),
        "write_fo": args.write_fo,
        "eval_tags": [EVAL_ERR, EVAL_BAD],
        "class_map": [{"id": model_id, "name": name} for model_id, name in class_map.entries],
        "class_groups": {str(k): v for k, v in (groups or {}).items()},
        "background_id": background_id,
        "device": str(device),
        "n_scored": stats.n_scored,
        "n_ok": stats.n_ok,
        "n_err": stats.n_err,
        "n_bad_tag": stats.n_bad,
        "n_no_gt_tag": stats.n_no_gt,
        "n_multi_gt_tag": stats.n_multi_gt,
        "err_rate": (stats.n_err / stats.n_scored) if stats.n_scored else 0.0,
        "by_gt": dict(stats.by_gt),
        "n_tagged": n_tagged,
        "n_problems": len(problems),
        "results_csv": str(results_csv),
        "problems_path": str(problems_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)

    print(
        f"scored={stats.n_scored} ok={stats.n_ok} err={stats.n_err} "
        f"bad_tag={stats.n_bad} no_gt={stats.n_no_gt} multi_gt={stats.n_multi_gt} "
        f"err_rate={summary['err_rate']:.4f}"
    )
    for name, bucket in sorted(stats.by_gt.items()):
        rate = (bucket["err"] / bucket["n"]) if bucket["n"] else 0.0
        print(f"  gt={name} n={bucket['n']} err={bucket['err']} err_rate={rate:.4f}")
    if args.write_fo:
        print(f"tagged={n_tagged}")
        print(f"app filter: detection tag `{EVAL_ERR}` or `{EVAL_BAD}`")
    print(f"results_csv={results_csv}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"problems_path={problems_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
