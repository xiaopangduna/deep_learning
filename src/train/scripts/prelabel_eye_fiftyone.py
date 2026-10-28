#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用框级分类器给 FiftyOne 里已有的头框预标眼睛开闭。

框来自数据集的检测字段（默认 ``ground_truth``），不读 CSV，不改类别名。
``--tag-map`` 把模型输出的训练组写成 ``eye_open_<n>``。没出现在映射里的组
（例如 background）不写标签。``eye_open_1`` 是人工也难分、仍归在闭眼的框，
除非映射里明确写出，否则脚本不生成这个标签。

已有 ``eye_open_*`` 的框默认跳过。眼睛标签实际发生变化的框会加上 ``changed``，
便于在 App 里按框复查。预测与原标签相同则不动。不加 ``--write`` 只出计划，不改库。
模型加载方式与 ``val_classifier_with_fiftyone.py`` 相同。

示例::

    # v001：0 是 background（不写），1 是闭眼，2 是开眼
    uv run scripts/prelabel_eye_fiftyone.py \\
        --ckpt logs/.../epoch=19-val_acc=0.96.ckpt \\
        --config configs/experiments/image_classifiter_from_detect_box_head_eye_open_mobilenet_v2_v001.yaml \\
        --dataset BBM08S_head \\
        --sample-tags relabel \\
        --box-labels head \\
        --tag-map 1:eye_open_0,2:eye_open_2 \\
        --min-conf 0.8 \\
        --write

    # v100 没有 background：0 是闭眼，1 是开眼
    # --tag-map 0:eye_open_0,1:eye_open_2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EYE_TAG_PREFIX = "eye_open_"
CONF_ATTRIBUTE = "pred_eye_conf"
CHANGED_TAG = "changed"
DEFAULT_EXCLUDE_TAGS = ("dup_repeat_drop", "dup_near_drop")


def reference():
    """加载验证脚本里的模型构造函数。不修改那个文件。"""
    name = "val_classifier_with_fiftyone"
    cached = sys.modules.get(name)
    if cached is not None and hasattr(cached, "load_module"):
        return cached
    path = Path(__file__).resolve().parent / "val_classifier_with_fiftyone.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def comma_names(value: str) -> list[str]:
    names = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one name")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("duplicate names are not allowed")
    return names


def comma_names_optional(value: str) -> list[str]:
    return [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return number


def is_eye_tag(tag: object) -> bool:
    if not isinstance(tag, str) or not tag.startswith(EYE_TAG_PREFIX):
        return False
    return tag[len(EYE_TAG_PREFIX) :].isdigit()


def eye_tags(tags: object) -> list[str]:
    return [str(tag) for tag in (tags or []) if is_eye_tag(tag)]


def parse_tag_map(text: str) -> dict[int, str]:
    """解析 ``1:eye_open_0,2:eye_open_2``。组号必须由调用方再对照 class_groups。"""
    parts = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("--tag-map 不能为空")
    mapping: dict[int, str] = {}
    for part in parts:
        group_text, sep, tag = part.partition(":")
        group_text = group_text.strip()
        tag = tag.strip()
        if not sep or not group_text or not tag:
            raise ValueError(f"--tag-map 项须为 组id:eye_open_N，收到 {part!r}")
        if not group_text.lstrip("-").isdigit():
            raise ValueError(f"--tag-map 的组 id 须为整数，收到 {group_text!r}")
        if not is_eye_tag(tag):
            raise ValueError(f"--tag-map 的标签须为 eye_open_<数字>，收到 {tag!r}")
        group_id = int(group_text)
        if group_id in mapping:
            raise ValueError(f"--tag-map 组 {group_id} 重复")
        mapping[group_id] = tag
    return mapping


def model_output_ids(groups: dict[int, list[int]] | None, background_id: int | None) -> set[int]:
    ids = set(groups or {})
    if background_id is not None:
        ids.add(int(background_id))
    return ids


def validate_tag_map(
    tag_map: dict[int, str],
    groups: dict[int, list[int]] | None,
    background_id: int | None,
) -> None:
    known = model_output_ids(groups, background_id)
    if not known:
        raise ValueError("config 的 class_groups 为空，无法核对 --tag-map")
    unknown = sorted(set(tag_map) - known)
    if unknown:
        raise ValueError(
            f"--tag-map 的组不在 class_groups 里: {unknown}；模型输出组为 {sorted(known)}"
        )


def fo_xywh_to_yolo_cxcywh(box: list[float]) -> tuple[float, float, float, float]:
    """FiftyOne 相对左上角 xywh → YOLO 相对 cxcywh。"""
    x, y, width, height = (float(value) for value in box)
    return x + width / 2.0, y + height / 2.0, width, height


def geometry_status(
    width: float,
    height: float,
    img_w: float,
    img_h: float,
    min_side_px: float,
) -> str | None:
    """与框级数据集的几何筛选一致。通过时返回 None。"""
    if width <= 0 or height <= 0 or img_w <= 0 or img_h <= 0:
        return "invalid_box"
    if min_side_px > 0 and min(width * img_w, height * img_h) < min_side_px:
        return "short_side"
    return None


def decide_label(
    *,
    pred_group_id: int,
    pred_conf: float,
    tag_map: dict[int, str],
    existing: object,
    min_conf: float,
    overwrite: bool,
) -> tuple[str, str | None]:
    """返回 ``(status, tag)``。已有眼睛标签且未要求覆盖时，保留原标签。"""
    if eye_tags(existing) and not overwrite:
        return "skip_existing", None
    tag = tag_map.get(int(pred_group_id))
    if tag is None:
        return "unmapped", None
    if float(pred_conf) < float(min_conf):
        return "low_conf", None
    if eye_tags(existing) == [tag]:
        return "same", tag
    return "write", tag


def replace_eye_tag(tags: object, new_tag: str) -> list[str]:
    """去掉已有 ``eye_open_*``，写入 ``new_tag``，其它 tag 保留。"""
    kept = [str(tag) for tag in (tags or []) if not is_eye_tag(tag)]
    kept.append(new_tag)
    return kept


def tags_after_write(tags: object, new_tag: str) -> list[str]:
    """写入眼睛标签，并给这个框打上 ``changed``。"""
    updated = replace_eye_tag(tags, new_tag)
    if CHANGED_TAG not in updated:
        updated.append(CHANGED_TAG)
    return updated


@dataclass
class BoxJob:
    sample_id: str
    detection_id: str
    filepath: str
    det_label: str
    cx: float
    cy: float
    w: float
    h: float
    img_w: int
    img_h: int
    existing_tags: tuple[str, ...]
    status: str | None = None


@dataclass
class BoxResult:
    job: BoxJob
    pred_group_id: int | None
    pred_conf: float | None
    status: str
    tag: str | None


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


def collect_boxes(
    dataset,
    *,
    sample_tags: list[str],
    exclude_tags: list[str],
    label_field: str,
    box_labels: set[str],
    min_side_px: float,
    problems: list[dict],
) -> list[BoxJob]:
    import fiftyone as fo

    if label_field not in dataset.get_field_schema():
        raise ValueError(f"数据集没有字段 {label_field}")
    field = dataset.get_field(label_field)
    if not issubclass(field.document_type, fo.Detections):
        raise ValueError(f"字段 {label_field} 不是 Detections")
    view = dataset.match_tags(sample_tags, bool=True, all=True)
    excluded = set(exclude_tags)
    jobs: list[BoxJob] = []
    for sample in view.iter_samples(progress=True, autosave=False):
        tags = {str(tag) for tag in (sample.tags or [])}
        if tags & excluded:
            continue
        size = image_size(sample)
        if size is None:
            problems.append(
                {
                    "sample_id": str(sample.id),
                    "file": str(sample.filepath),
                    "reason": "missing_image",
                }
            )
            continue
        img_w, img_h = size
        label = sample[label_field]
        detections = list(getattr(label, "detections", None) or [])
        for detection in detections:
            if str(detection.label) not in box_labels:
                continue
            box = list(getattr(detection, "bounding_box", None) or [])
            if len(box) != 4:
                problems.append(
                    {
                        "sample_id": str(sample.id),
                        "detection_id": str(getattr(detection, "id", "")),
                        "reason": "bad_box",
                    }
                )
                continue
            cx, cy, width, height = fo_xywh_to_yolo_cxcywh(box)
            status = geometry_status(width, height, img_w, img_h, min_side_px)
            jobs.append(
                BoxJob(
                    sample_id=str(sample.id),
                    detection_id=str(detection.id),
                    filepath=str(sample.filepath),
                    det_label=str(detection.label),
                    cx=cx,
                    cy=cy,
                    w=width,
                    h=height,
                    img_w=img_w,
                    img_h=img_h,
                    existing_tags=tuple(str(tag) for tag in (detection.tags or []) if isinstance(tag, str)),
                    status=status,
                )
            )
    return jobs


def write_predict_csv(path: Path, jobs: list[BoxJob]) -> None:
    pd.DataFrame(
        [
            {
                "path_img": job.filepath,
                "cx": job.cx,
                "cy": job.cy,
                "w": job.w,
                "h": job.h,
                "img_w": job.img_w,
                "img_h": job.img_h,
                "sample_id": job.sample_id,
                "detection_id": job.detection_id,
            }
            for job in jobs
        ]
    ).to_csv(path, index=False)


def build_predict_dataset(cfg: dict[str, Any], csv_path: Path):
    ref = reference()
    from lovely_deep_learning.dataset.image_classifier import ImageClassifierFromDetectBoxDataset

    init_args = ref.data_init_args(cfg)
    transform_spec = init_args.get("transform_val")
    if transform_spec is None:
        raise ValueError("config data.init_args.transform_val is required")
    return ImageClassifierFromDetectBoxDataset(
        csv_paths=[csv_path],
        key_map=dict(ImageClassifierFromDetectBoxDataset.PREDICT_KEY_MAP),
        transform=ref.instantiate_nested(transform_spec),
        norm_mean=init_args.get("norm_mean"),
        norm_std=init_args.get("norm_std"),
        box_scale=float(init_args.get("box_scale", 1.2)),
        min_side_px=float(init_args.get("min_side_px", 16.0)),
        class_groups=init_args.get("class_groups"),
        background_ratio=0.0,
        background_iou=float(init_args.get("background_iou", 0.3)),
    )


def run_inference(
    module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, tuple[int, float]]:
    from lovely_deep_learning.dataset.base import BaseDataset
    from lovely_deep_learning.dataset.image_classifier import ImageClassifierDataset

    collate = ImageClassifierDataset.get_collate_fn_for_dataloader()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    found: dict[str, tuple[int, float]] = {}
    offset = 0
    with torch.inference_mode():
        for net_in, _net_out in tqdm(loader, desc="infer", unit="batch"):
            images = BaseDataset.stack_batch_images(net_in).to(device, non_blocking=True)
            parsed = module.postprocess.run(module(images))
            pred_ids = parsed["pred_ids"].detach().cpu()
            pred_conf = parsed["pred_conf"].detach().cpu()
            batch_n = len(net_in)
            for index in range(batch_n):
                row = dataset.sample_path_table.iloc[offset + index]
                found[str(row["detection_id"])] = (
                    int(pred_ids[index].item()),
                    float(pred_conf[index].item()),
                )
            offset += batch_n
    return found


def label_jobs(
    jobs: list[BoxJob],
    predictions: dict[str, tuple[int, float]],
    *,
    tag_map: dict[int, str],
    min_conf: float,
    overwrite: bool,
) -> list[BoxResult]:
    results: list[BoxResult] = []
    for job in jobs:
        if job.status is not None:
            results.append(BoxResult(job, None, None, job.status, None))
            continue
        predicted = predictions.get(job.detection_id)
        if predicted is None:
            results.append(BoxResult(job, None, None, "filtered", None))
            continue
        group_id, confidence = predicted
        status, tag = decide_label(
            pred_group_id=group_id,
            pred_conf=confidence,
            tag_map=tag_map,
            existing=job.existing_tags,
            min_conf=min_conf,
            overwrite=overwrite,
        )
        results.append(BoxResult(job, group_id, confidence, status, tag))
    return results


def write_labels(dataset, results: list[BoxResult], label_field: str, problems: list[dict]) -> int:
    by_sample: dict[str, list[BoxResult]] = defaultdict(list)
    for result in results:
        if result.status == "write" and result.tag is not None:
            by_sample[result.job.sample_id].append(result)
    written = 0
    for sample_id, rows in tqdm(by_sample.items(), desc="write", unit="sample"):
        sample = dataset[sample_id]
        label = sample[label_field]
        detections = {str(detection.id): detection for detection in (label.detections or [])}
        changed = False
        for result in rows:
            detection = detections.get(result.job.detection_id)
            if detection is None or result.tag is None or result.pred_conf is None:
                problems.append(
                    {
                        "sample_id": sample_id,
                        "detection_id": result.job.detection_id,
                        "reason": "missing_detection",
                    }
                )
                result.status = "missing_detection"
                result.tag = None
                continue
            detection.tags = tags_after_write(detection.tags, result.tag)
            detection.set_attribute_value(CONF_ATTRIBUTE, round(result.pred_conf, 6))
            changed = True
            written += 1
        if changed:
            sample[label_field] = label
            sample.save()
    return written


def result_row(result: BoxResult) -> dict[str, object]:
    job = result.job
    return {
        "sample_id": job.sample_id,
        "detection_id": job.detection_id,
        "filepath": job.filepath,
        "det_label": job.det_label,
        "cx": job.cx,
        "cy": job.cy,
        "w": job.w,
        "h": job.h,
        "existing_eye": ",".join(eye_tags(job.existing_tags)),
        "pred_group_id": result.pred_group_id,
        "pred_conf": result.pred_conf,
        "tag": result.tag,
        "status": result.status,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, type=Path, help="Lightning .ckpt")
    parser.add_argument("--config", required=True, type=Path, help="与 checkpoint 分类头一致的实验 YAML")
    parser.add_argument("--dataset", required=True, help="FiftyOne 数据集名")
    parser.add_argument("--sample-tags", required=True, type=comma_names, help="样本须同时带有这些 tag")
    parser.add_argument("--box-labels", required=True, type=comma_names, help="要分类的 detection.label，例如 head")
    parser.add_argument(
        "--tag-map",
        required=True,
        help="训练组到眼睛标签，例如 1:eye_open_0,2:eye_open_2。未列出的组不标注",
    )
    parser.add_argument("--label-field", default="ground_truth", help="检测框字段，默认 ground_truth")
    parser.add_argument(
        "--exclude-tags",
        type=comma_names_optional,
        default=list(DEFAULT_EXCLUDE_TAGS),
        help="带其中任一 tag 的样本跳过。默认 dup_repeat_drop,dup_near_drop",
    )
    parser.add_argument("--min-conf", type=unit_interval, default=0.0, help="低于该置信度不写标签")
    parser.add_argument("--overwrite", action="store_true", help="替换框上已有的 eye_open_*")
    parser.add_argument("--write", action="store_true", help="写回 FiftyOne。默认只出计划")
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0，默认自动")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="计划与摘要目录")
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
    try:
        tag_map = parse_tag_map(args.tag_map)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    ref = reference()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        module, cfg = ref.load_module(config_path, ckpt_path, device)
        init_args = ref.data_init_args(cfg)
        groups, background_id = ref.parse_class_groups_from_cfg(init_args)
        validate_tag_map(tag_map, groups, background_id)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    ref.ensure_fiftyone_database_uri()
    import fiftyone as fo

    if not fo.dataset_exists(args.dataset):
        print(f"error: FiftyOne dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    dataset = fo.load_dataset(args.dataset)
    min_side_px = float(init_args.get("min_side_px", 16.0))
    batch_size = int(args.batch_size or init_args.get("batch_size") or 32)
    num_workers = int(args.num_workers if args.num_workers is not None else init_args.get("num_workers") or 0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out_dir or (ROOT / "logs" / "prelabel_eye_fo" / stamp)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"ckpt={ckpt_path}")
    print(f"config={config_path}")
    print(f"dataset={args.dataset} label_field={args.label_field}")
    print(f"sample_tags={args.sample_tags} exclude_tags={args.exclude_tags}")
    print(f"box_labels={args.box_labels}")
    print(f"class_groups={groups} background_id={background_id}")
    print(f"tag_map={tag_map} min_conf={args.min_conf} overwrite={args.overwrite} write={args.write}")

    problems: list[dict] = []
    try:
        jobs = collect_boxes(
            dataset,
            sample_tags=args.sample_tags,
            exclude_tags=args.exclude_tags,
            label_field=args.label_field,
            box_labels=set(args.box_labels),
            min_side_px=min_side_px,
            problems=problems,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not jobs and not any(item.get("reason") == "missing_image" for item in problems):
        print("error: 没有样本同时带有 --sample-tags，或这些样本没有可标的框", file=sys.stderr)
        return 2

    infer_jobs = [job for job in jobs if job.status is None]
    predictions: dict[str, tuple[int, float]] = {}
    if infer_jobs:
        csv_path = out_dir / "predict_boxes.csv"
        write_predict_csv(csv_path, infer_jobs)
        predict_dataset = build_predict_dataset(cfg, csv_path)
        predictions = run_inference(
            module,
            predict_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    results = label_jobs(
        jobs,
        predictions,
        tag_map=tag_map,
        min_conf=args.min_conf,
        overwrite=args.overwrite,
    )
    written = 0
    if args.write:
        written = write_labels(dataset, results, args.label_field, problems)

    counts = Counter(result.status for result in results)
    report_path = out_dir / "prelabel.csv"
    pd.DataFrame([result_row(result) for result in results]).to_csv(report_path, index=False)
    problems_path = out_dir / "problems.jsonl"
    with problems_path.open("w", encoding="utf-8") as handle:
        for item in problems:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "ckpt": str(ckpt_path),
        "config": str(config_path),
        "dataset": args.dataset,
        "label_field": args.label_field,
        "sample_tags": args.sample_tags,
        "exclude_tags": args.exclude_tags,
        "box_labels": args.box_labels,
        "tag_map": {str(key): value for key, value in sorted(tag_map.items())},
        "class_groups": {str(key): value for key, value in (groups or {}).items()},
        "background_id": background_id,
        "min_conf": args.min_conf,
        "overwrite": args.overwrite,
        "write": args.write,
        "n_boxes": len(results),
        "n_written": written,
        "counts": dict(counts),
        "n_problems": len(problems),
        "report": str(report_path),
        "problems_path": str(problems_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)
    print(
        " ".join(
            [
                f"boxes={len(results)}",
                f"write={counts.get('write', 0)}",
                f"written={written}",
                f"same={counts.get('same', 0)}",
                f"skip_existing={counts.get('skip_existing', 0)}",
                f"unmapped={counts.get('unmapped', 0)}",
                f"low_conf={counts.get('low_conf', 0)}",
                f"short_side={counts.get('short_side', 0)}",
            ]
        )
    )
    print(f"report={report_path}")
    print(f"summary={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
