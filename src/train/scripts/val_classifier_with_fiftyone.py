#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 GT 框跑分类器，可选写入 FiftyOne（``pred_eye_<ver>`` + OK/ERR tags）。

样本用原图 SHA-256 对齐数据集 ``sha256`` 字段。默认只打分；``--write-fo --ver``
才改库。``--care-classes`` 指定关注类时，ERR 按「是否属于 care」的 XOR 判定
（如只关心 ``eye_open_2``）；未指定则在训练组空间一一对应。

示例::

    # 只打分（不写 FO）
    uv run scripts/val_classifier_with_fiftyone.py \\
        --ckpt logs/image_classifiter_from_detect_box_head_eye_open/version_6/checkpoints/epoch=19-val_acc=0.96.ckpt \\
        --config configs/experiments/image_classifiter_from_detect_box_head_eye_open_v100.yaml \\
        --csv datasets/head_eye_open/test_v301.csv \\
        --care-classes eye_open_2

    # 写入 BBM08S_head（本机已有 mongod 时默认连 27017）
    export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017   # 可选
    uv run scripts/val_classifier_with_fiftyone.py \\
        --ckpt logs/.../epoch=19-val_acc=0.96.ckpt \\
        --config configs/experiments/image_classifiter_from_detect_box_head_eye_open_v100.yaml \\
        --csv datasets/head_eye_open/test_v301.csv \\
        --dataset BBM08S_head \\
        --care-classes eye_open_2 \\
        --write-fo --ver v100
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
SHA256_CHUNK_SIZE = 1024 * 1024
VERSION_DET_TAGS = ("OK", "ERR")


@dataclass
class BoxPrediction:
    """One GT-box classification result."""

    img_path: Path
    sha256: str
    cx: float
    cy: float
    w: float
    h: float
    gt_original_id: int
    gt_original_name: str
    gt_group_id: int
    pred_group_id: int
    pred_group_name: str
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


def parse_ver(value: str) -> str:
    stripped = value.strip()
    if not stripped or "/" in stripped or "\\" in stripped or stripped in {".", ".."}:
        raise argparse.ArgumentTypeError("--ver must be a short token like v100")
    if stripped.startswith("pred_"):
        raise argparse.ArgumentTypeError("--ver should be like v100 (pred_eye_ prefix is added automatically)")
    return stripped


def compute_sha256_hex(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pred_field_name(ver: str) -> str:
    return f"pred_eye_{ver}"


def det_tag(ver: str, kind: str) -> str:
    return f"{ver}_{kind}"


def sample_tag_ok(ver: str) -> str:
    return f"pred_eye_{ver}_ok"


def sample_tag_bad(ver: str) -> str:
    return f"pred_eye_{ver}_bad"


def yolo_cxcywh_to_fo_box(cx: float, cy: float, w: float, h: float) -> list[float]:
    """YOLO relative cxcywh → FiftyOne relative top-left xywh."""
    return [float(cx) - float(w) / 2.0, float(cy) - float(h) / 2.0, float(w), float(h)]


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


def build_sha256_index(dataset) -> tuple[dict[str, list[str]], int]:
    ids, hashes = dataset.values(["id", "sha256"])
    index: dict[str, list[str]] = defaultdict(list)
    missing = 0
    for sample_id, digest in zip(ids, hashes):
        if not digest:
            missing += 1
            continue
        index[str(digest)].append(str(sample_id))
    return dict(index), missing


def update_sample_tags(sample, ver: str, is_bad: bool) -> None:
    ok, bad = sample_tag_ok(ver), sample_tag_bad(ver)
    tags = [tag for tag in (sample.tags or []) if tag not in {ok, bad}]
    tags.append(bad if is_bad else ok)
    sample.tags = tags


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
    problems: list[dict],
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
    sha_cache: dict[Path, str] = {}
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
                try:
                    digest = sha_cache.get(img_path)
                    if digest is None:
                        digest = compute_sha256_hex(img_path)
                        sha_cache[img_path] = digest
                except OSError as error:
                    append_problem(
                        problems,
                        file=str(img_path),
                        reason="read_error",
                        detail=str(error),
                        row=index,
                    )
                    continue
                table_row = dataset.sample_path_table.iloc[index]
                gt_original_id = int(dataset._original_class_ids[index])
                gt_original_name = str(dataset._original_class_names[index])
                gt_group_id = int(net_out["class_id"][i].item()) if net_out else int(
                    original_to_group.get(gt_original_id, -1)
                )
                pred_group_id = int(pred_ids[i].item())
                pred_group_name = dataset.class_name_for_id(pred_group_id)
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
                        img_path=img_path,
                        sha256=digest,
                        cx=float(table_row["cx"]),
                        cy=float(table_row["cy"]),
                        w=float(table_row["w"]),
                        h=float(table_row["h"]),
                        gt_original_id=gt_original_id,
                        gt_original_name=gt_original_name,
                        gt_group_id=gt_group_id,
                        pred_group_id=pred_group_id,
                        pred_group_name=pred_group_name,
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
    dataset_name: str,
    rows: list[BoxPrediction],
    ver: str,
    problems: list[dict],
) -> tuple[int, int, int, int]:
    ensure_fiftyone_database_uri()
    import fiftyone as fo

    if dataset_name not in fo.list_datasets():
        raise SystemExit(f"error: FiftyOne dataset not found: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    hash_index, missing_hash_samples = build_sha256_index(dataset)
    pred_field = pred_field_name(ver)
    tag_ok = det_tag(ver, "OK")
    tag_err = det_tag(ver, "ERR")

    by_sample: dict[str, list[BoxPrediction]] = defaultdict(list)
    seen_paths: dict[str, str] = {}
    for row in rows:
        candidates = hash_index.get(row.sha256, [])
        if not candidates:
            append_problem(
                problems,
                file=str(row.img_path),
                sha256=row.sha256,
                reason="not_in_fiftyone",
            )
            continue
        if len(candidates) > 1:
            append_problem(
                problems,
                file=str(row.img_path),
                sha256=row.sha256,
                reason="ambiguous_sha256",
                sample_ids=candidates,
            )
            continue
        sample_id = candidates[0]
        prev = seen_paths.get(row.sha256)
        if prev is not None and prev != str(row.img_path):
            append_problem(
                problems,
                file=str(row.img_path),
                sha256=row.sha256,
                reason="sha256_path_conflict",
                other_file=prev,
            )
            continue
        seen_paths[row.sha256] = str(row.img_path)
        by_sample[sample_id].append(row)

    n_written = n_ok = n_bad = 0
    for sample_id, sample_rows in tqdm(by_sample.items(), desc="write-fo", unit="sample"):
        sample = dataset[sample_id]
        detections = []
        any_err = False
        for row in sample_rows:
            kind = tag_err if row.is_err else tag_ok
            any_err = any_err or row.is_err
            det = fo.Detection(
                label=row.pred_group_name,
                bounding_box=yolo_cxcywh_to_fo_box(row.cx, row.cy, row.w, row.h),
                confidence=round(row.pred_conf, 6),
                tags=[kind],
            )
            det["gt_original_name"] = row.gt_original_name
            det["gt_original_id"] = int(row.gt_original_id)
            det["pred_group_id"] = int(row.pred_group_id)
            detections.append(det)
        sample[pred_field] = fo.Detections(detections=detections)
        update_sample_tags(sample, ver, any_err)
        sample.save()
        n_written += 1
        if any_err:
            n_bad += 1
        else:
            n_ok += 1
    print(f"dataset_samples_missing_sha256={missing_hash_samples}")
    return n_written, n_ok, n_bad, missing_hash_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, type=Path, help="Lightning .ckpt path")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML (must match ckpt head)")
    parser.add_argument("--csv", required=True, type=Path, help="Box-crop CSV with labels")
    parser.add_argument("--dataset", type=nonempty, default=None, help="FiftyOne dataset name")
    parser.add_argument(
        "--care-classes",
        default=None,
        help="Comma-separated focus classes (id / name / eye_open_2). Omit for exact group match.",
    )
    parser.add_argument("--write-fo", action="store_true", help="Write pred_eye_<ver> into FiftyOne")
    parser.add_argument("--ver", type=parse_ver, default=None, help="Version token, required with --write-fo")
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
    if args.write_fo and not args.ver:
        print("error: --write-fo requires --ver (e.g. v100)", file=sys.stderr)
        return 2
    if args.write_fo and not args.dataset:
        print("error: --write-fo requires --dataset", file=sys.stderr)
        return 2

    ckpt_path = args.ckpt.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"error: ckpt not found: {ckpt_path}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2
    if not csv_path.is_file():
        print(f"error: csv not found: {csv_path}", file=sys.stderr)
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module, cfg = load_module(config_path, ckpt_path, device)
    init_args = data_init_args(cfg)
    groups, background_id = parse_class_groups_from_cfg(init_args)
    dataset = build_dataset(cfg, csv_path)
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out_dir or (ROOT / "logs" / "val_classifier_fo" / stamp)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    problems: list[dict] = []
    print(f"device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"config={config_path}")
    print(f"csv={csv_path} boxes={len(dataset)}")
    print(f"class_groups={groups} background_id={background_id}")
    print(f"care_classes={sorted(care_ids) if care_ids is not None else None}")
    print(f"write_fo={args.write_fo} ver={args.ver} dataset={args.dataset}")

    rows = run_inference(
        module,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        care_ids=care_ids,
        problems=problems,
    )
    stats = EvalStats()
    for row in rows:
        stats.add(row)

    results_csv = out_dir / "box_results.csv"
    pd.DataFrame(
        [
            {
                "img_path": str(row.img_path),
                "sha256": row.sha256,
                "cx": row.cx,
                "cy": row.cy,
                "w": row.w,
                "h": row.h,
                "gt_original_id": row.gt_original_id,
                "gt_original_name": row.gt_original_name,
                "gt_group_id": row.gt_group_id,
                "pred_group_id": row.pred_group_id,
                "pred_group_name": row.pred_group_name,
                "pred_conf": row.pred_conf,
                "is_err": row.is_err,
            }
            for row in rows
        ]
    ).to_csv(results_csv, index=False)

    n_written = n_ok = n_bad = missing_hash = 0
    if args.write_fo:
        assert args.ver is not None and args.dataset is not None
        try:
            n_written, n_ok, n_bad, missing_hash = write_fiftyone(
                dataset_name=args.dataset,
                rows=rows,
                ver=args.ver,
                problems=problems,
            )
        except ImportError:
            print(
                "error: fiftyone is not installed in this environment; "
                "install it or run scoring without --write-fo",
                file=sys.stderr,
            )
            return 2

    problems_path = out_dir / "problems.jsonl"
    with problems_path.open("w", encoding="utf-8") as handle:
        for item in problems:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "ckpt": str(ckpt_path),
        "config": str(config_path),
        "csv": str(csv_path),
        "dataset": args.dataset,
        "write_fo": args.write_fo,
        "ver": args.ver,
        "pred_field": pred_field_name(args.ver) if args.ver else None,
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
        "n_written": n_written,
        "n_sample_ok": n_ok,
        "n_sample_bad": n_bad,
        "dataset_samples_missing_sha256": missing_hash,
        "n_problems": len(problems),
        "sample_tag_ok": sample_tag_ok(args.ver) if args.ver else None,
        "sample_tag_bad": sample_tag_bad(args.ver) if args.ver else None,
        "det_tags": [det_tag(args.ver, kind) for kind in VERSION_DET_TAGS] if args.ver else [],
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
        print(f"written={n_written} sample_ok={n_ok} sample_bad={n_bad}")
        print(f"pred_field={pred_field_name(args.ver)}")
        print(f"app filter: sample tag `{sample_tag_bad(args.ver)}` or detection tag `{det_tag(args.ver, 'ERR')}`")
    print(f"results_csv={results_csv}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"problems_path={problems_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
