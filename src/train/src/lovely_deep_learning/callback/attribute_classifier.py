"""多属性分类的可视化。

单类分类只在脸上写一行类名。这里每个属性都是独立的 0/1，40 行字盖住脸，
所以只列出真值或预测为 1 的属性，其余真负例不画。

- ``LogAttributeClassifierValVisualizationCallback``：验证集前几张脸，以及
  全部验证样本的逐属性准确率 / 召回率，写入 TensorBoard。
- ``SaveAttributeClassifierTestPredictVisualizationCallback``：测试时按属性
  保存误报和漏报，并写每张图的概率。预测有标签时同样对照真值；没有标签时只标出预测为 1 的属性。

训练图不保存。准确率对稀有属性很高（全预测 0 也能到 90% 以上），看召回率和
误报 / 漏报图才能判断属性头是否正常。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from ..dataset.base import BaseDataset
from ..dataset.image_classifier import ImageClassifierDataset

COLOR_TP: tuple[int, int, int] = (0, 170, 0)
COLOR_FP: tuple[int, int, int] = (0, 0, 220)
COLOR_FN: tuple[int, int, int] = (0, 140, 255)
COLOR_PRED: tuple[int, int, int] = (220, 120, 0)
COLOR_NONE: tuple[int, int, int] = (120, 120, 120)

_KIND_COLOR = {
    "tp": COLOR_TP,
    "fp": COLOR_FP,
    "fn": COLOR_FN,
    "pred": COLOR_PRED,
}
_KIND_ORDER = {"fp": 0, "fn": 1, "tp": 2, "pred": 3}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def attribute_display_name(column: str) -> str:
    """去掉 CSV 列名的 ``attr_`` 前缀。"""
    text = str(column)
    if text.startswith("attr_"):
        return text[len("attr_") :]
    return text


def visible_attribute_rows(
    names: list[str],
    probs: torch.Tensor | np.ndarray,
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray | None = None,
) -> list[tuple[str, float, str]]:
    """返回要画出来的 ``(名字, 概率, 种类)``。

    种类是 ``tp`` / ``fp`` / ``fn``；没有真值时，预测为 1 的是 ``pred``。
    真值和预测都是 0 的属性丢掉。误报、漏报排在前面，同一种类里概率高的在前。
    """
    pred_b = _as_numpy_binary(pred)
    prob_v = np.asarray(probs, dtype=np.float64).reshape(-1)
    target_b = None if target is None else _as_numpy_binary(target)
    if pred_b.shape[0] != len(names) or prob_v.shape[0] != len(names):
        raise ValueError(
            f"属性名 {len(names)} 个，预测 {pred_b.shape[0]} 个，概率 {prob_v.shape[0]} 个"
        )
    if target_b is not None and target_b.shape != pred_b.shape:
        raise ValueError(f"真值 {target_b.shape} 与预测 {pred_b.shape} 形状不一致")

    rows: list[tuple[str, float, str]] = []
    for index, name in enumerate(names):
        turned_on = bool(pred_b[index])
        probability = float(prob_v[index])
        if target_b is None:
            if turned_on:
                rows.append((name, probability, "pred"))
            continue
        labeled = bool(target_b[index])
        if not turned_on and not labeled:
            continue
        if turned_on and labeled:
            kind = "tp"
        elif turned_on:
            kind = "fp"
        else:
            kind = "fn"
        rows.append((name, probability, kind))
    rows.sort(key=lambda item: (_KIND_ORDER[item[2]], -item[1], item[0]))
    return rows


def render_attribute_panel(
    face_bgr: np.ndarray,
    rows: list[tuple[str, float, str]],
    *,
    with_target: bool,
) -> np.ndarray:
    """左边是脸，右边是属性。每种结果前面有一块纯色，便于和文字颜色对照。"""
    face = np.ascontiguousarray(face_bgr)
    if face.ndim != 3 or face.shape[2] != 3:
        raise ValueError(f"脸图须为 HWC BGR，收到 {face.shape}")

    scale = 0.4
    thickness = 1
    line_h = 16
    pad = 6
    swatch = 8
    legend = [("both", "tp"), ("pred-only", "fp"), ("gt-only", "fn")] if with_target else [("pred", "pred")]
    lines = [(f"{name}  {prob:.2f}", kind) for name, prob, kind in rows]
    if not lines:
        lines = [("all negative", "none")]

    def text_width(text: str) -> int:
        (width, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        return int(width)

    content_w = max(text_width(text) for text, _ in lines)
    content_w = max(content_w, sum(text_width(text) + swatch + 12 for text, _ in legend))
    panel_w = content_w + pad * 2 + swatch + 6
    panel_h = max(face.shape[0], pad + line_h + line_h * len(lines) + pad)
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    cursor_x = pad
    legend_y = pad + 11
    for text, kind in legend:
        _swatch(panel, cursor_x, legend_y - 8, _KIND_COLOR[kind])
        cursor_x += swatch + 4
        cv2.putText(panel, text, (cursor_x, legend_y), _FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cursor_x += text_width(text) + 10

    y = legend_y + line_h
    for text, kind in lines:
        color = COLOR_NONE if kind == "none" else _KIND_COLOR[kind]
        _swatch(panel, pad, y - 8, color)
        cv2.putText(
            panel,
            text,
            (pad + swatch + 4, y),
            _FONT,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    if face.shape[0] < panel_h:
        padded = np.full((panel_h, face.shape[1], 3), 255, dtype=np.uint8)
        padded[: face.shape[0]] = face
        face = padded
    gap = np.full((panel_h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([face, gap, panel], axis=1)


def render_accuracy_recall_chart(
    names: list[str],
    accuracy: np.ndarray,
    recall: np.ndarray,
) -> np.ndarray:
    """两行条形图，返回 RGB uint8。召回率没有正样本的位置留空。"""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [attribute_display_name(name) for name in names]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.32 * len(labels) + 1.5), 7.0))
    axes[0].bar(x, np.asarray(accuracy, dtype=np.float64), color="#4C78A8")
    axes[1].bar(x, np.asarray(recall, dtype=np.float64), color="#F58518")
    axes[0].set_title("accuracy")
    axes[1].set_title("recall")
    for axis in axes:
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class AttributeCount:
    """逐属性累计正确数、真正例和真值为 1 的个数。"""

    def __init__(self) -> None:
        self.correct: torch.Tensor | None = None
        self.tp: torch.Tensor | None = None
        self.gt_pos: torch.Tensor | None = None
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_b = _as_binary_tensor(pred)
        target_b = _as_binary_tensor(target)
        if pred_b.shape != target_b.shape:
            raise ValueError(f"预测 {tuple(pred_b.shape)} 与真值 {tuple(target_b.shape)} 形状不一致")
        correct = pred_b.eq(target_b).sum(dim=0).to(torch.int64)
        tp = (pred_b & target_b).sum(dim=0).to(torch.int64)
        positive = target_b.sum(dim=0).to(torch.int64)
        if self.correct is None:
            self.correct = correct
            self.tp = tp
            self.gt_pos = positive
        elif self.correct.shape != correct.shape:
            raise ValueError(
                f"属性数从 {self.correct.shape[0]} 变成 {correct.shape[0]}"
            )
        else:
            self.correct += correct
            self.tp += tp
            self.gt_pos += positive
        self.count += int(pred_b.shape[0])

    def accuracy_recall(self) -> tuple[np.ndarray, np.ndarray]:
        if self.correct is None or self.count == 0:
            raise RuntimeError("还没有样本")
        accuracy = (self.correct.float() / self.count).numpy()
        recall = np.full(self.correct.shape[0], np.nan, dtype=np.float64)
        positive = self.gt_pos
        assert positive is not None and self.tp is not None
        mask = positive > 0
        recall[mask.numpy()] = (self.tp[mask].float() / positive[mask].float()).numpy()
        return accuracy, recall


class LogAttributeClassifierValVisualizationCallback(pl.Callback):
    """每个验证 epoch 把前几张脸和逐属性准确率 / 召回率写入 TensorBoard。

    脸来自该 epoch 第一个 batch 的前 ``max_images`` 张。验证集不打乱时，
    这就是 CSV 里的前几行，方便逐 epoch 看同一批脸。条形图用本进程看到的
    全部验证 batch。单卡时这是完整验证集。
    """

    def __init__(self, max_images: int = 8, nrow: int = 2) -> None:
        super().__init__()
        if max_images <= 0:
            raise ValueError(f"max_images 须为正数，收到 {max_images}")
        if nrow <= 0:
            raise ValueError(f"nrow 须为正数，收到 {nrow}")
        self.max_images = int(max_images)
        self.nrow = int(nrow)
        self._counts = AttributeCount()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._counts = AttributeCount()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0 or _skip_rank_or_sanity(trainer):
            return
        if not isinstance(outputs, dict):
            return
        dataset = _val_dataset(trainer)
        parsed = _parse_batch(dataset, batch, outputs)
        if parsed is None:
            return
        names, net_in, probs, pred, target = parsed
        if target is None:
            return
        try:
            self._counts.update(pred, target)
            if batch_idx != 0:
                return
            display = [attribute_display_name(name) for name in names]
            panels = []
            for index in range(min(self.max_images, pred.shape[0])):
                rows = visible_attribute_rows(display, probs[index], pred[index], target[index])
                face = _face_bgr(dataset, net_in[index])
                panels.append(render_attribute_panel(face, rows, with_target=True))
            if not panels:
                return
            _add_image(trainer, pl_module, "val/sample_attributes", _panel_grid(panels, self.nrow))
        except Exception as exc:
            print(
                "Warning: failed to log val attribute visualization at step "
                f"{getattr(pl_module, 'global_step', 0)}, {exc}"
            )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if _skip_rank_or_sanity(trainer) or self._counts.count == 0 or self._counts.correct is None:
            return
        dataset = _val_dataset(trainer)
        names = _column_names(dataset, int(self._counts.correct.shape[0]))
        if names is None:
            return
        try:
            accuracy, recall = self._counts.accuracy_recall()
            chart = render_accuracy_recall_chart(names, accuracy, recall)
            _add_image(trainer, pl_module, "val/attribute_accuracy_recall", chart)
        except Exception as exc:
            print(
                "Warning: failed to log val attribute metrics at step "
                f"{getattr(pl_module, 'global_step', 0)}, {exc}"
            )


class SaveAttributeClassifierTestPredictVisualizationCallback(pl.Callback):
    """测试写误报 / 漏报图和概率表。

    预测有标签时和测试一样用绿 / 红 / 橙对照，但保存的是图片而不是只存错例。
    没有标签时只画预测为 1 的属性。

    ``save_dir`` 为空时用本次 ``trainer.log_dir``。每个属性的误报、漏报
    各自最多 ``max_per_bucket`` 张，按 dataloader 顺序取前面的。概率 CSV
    覆盖全部测试图。``max_predict_images`` 为空时预测图全部保存。

    目录::

        {root}/test/{属性}/fp/*.jpg
        {root}/test/{属性}/fn/*.jpg
        {root}/test_attribute_probs.csv
        {root}/per_attribute_metrics.csv
        {root}/per_attribute_metrics.png
        {root}/predict/*.jpg
        {root}/prediction_attribute_probs.csv
    """

    def __init__(
        self,
        save_dir: str | None = None,
        max_per_bucket: int = 8,
        max_predict_images: int | None = None,
    ) -> None:
        super().__init__()
        if max_per_bucket < 0:
            raise ValueError(f"max_per_bucket 不能为负，收到 {max_per_bucket}")
        if max_predict_images is not None and max_predict_images < 0:
            raise ValueError(f"max_predict_images 不能为负，收到 {max_predict_images}")
        self.max_per_bucket = int(max_per_bucket)
        self.max_predict_images = None if max_predict_images is None else int(max_predict_images)
        self._configured_save_dir = Path(save_dir) if save_dir else None
        self._reset_run_state()

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._reset_run_state()

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self._reset_run_state()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if dataloader_idx != 0 or not _is_rank_zero(trainer) or not isinstance(outputs, dict):
            return
        dataset = _loader_dataset(getattr(trainer, "test_dataloaders", None))
        parsed = _parse_batch(dataset, batch, outputs)
        if parsed is None:
            return
        names, net_in, probs, pred, target = parsed
        if target is None:
            print("Warning: test batch has no attribute labels, skip error images")
            return
        self._counts.update(pred, target)
        root = self._root(trainer)
        display = [attribute_display_name(name) for name in names]
        for index in range(pred.shape[0]):
            image_path = str(net_in[index].get("img_path", f"sample_{index}"))
            self._rows.append(_result_row(image_path, display, probs[index], pred[index], target[index]))
            if root is None:
                continue
            needed = self._open_error_buckets(pred[index], target[index])
            if not needed:
                continue
            rows = visible_attribute_rows(display, probs[index], pred[index], target[index])
            panel = render_attribute_panel(_face_bgr(dataset, net_in[index]), rows, with_target=True)
            stem = Path(image_path).stem
            for attr_index, kind in needed:
                directory = root / "test" / _safe_name(display[attr_index]) / kind
                cv2.imwrite(str(_jpg_path(directory, stem)), panel)
                self._saved[(attr_index, kind)] = self._saved.get((attr_index, kind), 0) + 1

    def on_test_end(self, trainer, pl_module) -> None:
        if not _is_rank_zero(trainer):
            return
        root = self._root(trainer)
        if root is None or not self._rows:
            return
        csv_path = root / "test_attribute_probs.csv"
        pd.DataFrame(self._rows).to_csv(csv_path, index=False)
        print(f"属性概率已保存到: {csv_path}")
        if self._counts.count == 0 or self._counts.correct is None:
            return
        dataset = _loader_dataset(getattr(trainer, "test_dataloaders", None))
        names = _column_names(dataset, int(self._counts.correct.shape[0]))
        if names is None:
            return
        accuracy, recall = self._counts.accuracy_recall()
        metrics_path = root / "per_attribute_metrics.csv"
        _write_metrics_csv(metrics_path, names, accuracy, recall, self._counts)
        chart = render_accuracy_recall_chart(names, accuracy, recall)
        chart_path = root / "per_attribute_metrics.png"
        cv2.imwrite(str(chart_path), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR))
        _add_image(trainer, pl_module, "test/attribute_accuracy_recall", chart)
        print(f"逐属性指标已保存到: {metrics_path}")

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0 or not _is_rank_zero(trainer) or not isinstance(outputs, dict):
            return
        dataset = _loader_dataset(getattr(trainer, "predict_dataloaders", None))
        parsed = _parse_batch(dataset, batch, outputs)
        if parsed is None:
            return
        names, net_in, probs, pred, target = parsed
        root = self._root(trainer)
        display = [attribute_display_name(name) for name in names]
        for index in range(pred.shape[0]):
            image_path = str(net_in[index].get("img_path", f"sample_{index}"))
            target_row = None if target is None else target[index]
            self._rows.append(_result_row(image_path, display, probs[index], pred[index], target_row))
            if root is None:
                continue
            if self.max_predict_images is not None and self._predict_saved >= self.max_predict_images:
                continue
            rows = visible_attribute_rows(display, probs[index], pred[index], target_row)
            panel = render_attribute_panel(
                _face_bgr(dataset, net_in[index]),
                rows,
                with_target=target_row is not None,
            )
            cv2.imwrite(str(_jpg_path(root / "predict", Path(image_path).stem)), panel)
            self._predict_saved += 1

    def on_predict_end(self, trainer, pl_module) -> None:
        if not _is_rank_zero(trainer):
            return
        root = self._root(trainer)
        if root is None or not self._rows:
            return
        csv_path = root / "prediction_attribute_probs.csv"
        pd.DataFrame(self._rows).to_csv(csv_path, index=False)
        print(f"预测概率已保存到: {csv_path}")

    def _reset_run_state(self) -> None:
        self._rows = []
        self._counts = AttributeCount()
        self._saved: dict[tuple[int, str], int] = {}
        self._predict_saved = 0
        self._root_ready = False
        self._root_path: Path | None = None

    def _root(self, trainer) -> Path | None:
        if self._root_ready:
            return self._root_path
        root = self._configured_save_dir
        if root is None:
            root = getattr(trainer, "log_dir", None)
            if not root:
                logger = getattr(trainer, "logger", None)
                root = getattr(logger, "log_dir", None) if logger is not None else None
        self._root_path = Path(root) if root else None
        self._root_ready = True
        if self._root_path is not None:
            self._root_path.mkdir(parents=True, exist_ok=True)
        return self._root_path

    def _open_error_buckets(self, pred_row: torch.Tensor, target_row: torch.Tensor) -> list[tuple[int, str]]:
        pred_b = _as_numpy_binary(pred_row)
        target_b = _as_numpy_binary(target_row)
        needed = []
        for index, (turned_on, labeled) in enumerate(zip(pred_b, target_b)):
            if bool(turned_on) == bool(labeled):
                continue
            kind = "fp" if turned_on else "fn"
            if self._saved.get((index, kind), 0) < self.max_per_bucket:
                needed.append((index, kind))
        return needed


def _parse_batch(dataset, batch, outputs):
    metric_preds = outputs.get("metric_preds")
    if dataset is None or not isinstance(metric_preds, dict):
        return None
    if "probs" not in metric_preds or "pred" not in metric_preds:
        return None
    net_in, net_out = batch
    if len(net_in) == 0:
        return None
    probs = metric_preds["probs"].detach().cpu()
    pred = metric_preds["pred"].detach().cpu()
    target = None
    if isinstance(net_out, dict) and "attr" in net_out:
        target = net_out["attr"].detach().cpu()
    width = int(pred.shape[1])
    names = _column_names(dataset, width)
    if names is None or probs.shape != pred.shape:
        print(f"Warning: attribute visualization skipped, names and pred width differ ({width})")
        return None
    if target is not None and tuple(target.shape) != tuple(pred.shape):
        print(f"Warning: attribute visualization skipped, target {tuple(target.shape)} pred {tuple(pred.shape)}")
        return None
    return names, net_in, probs, pred, target


def _face_bgr(dataset, sample: dict[str, Any]) -> np.ndarray:
    return ImageClassifierDataset.convert_img_from_tensor_to_numpy(dataset, _sample_image(sample))


def _column_names(dataset, width: int) -> list[str] | None:
    columns = list(getattr(dataset, "attribute_columns", []) or [])
    if len(columns) != width:
        return None
    return columns


def _sample_image(sample: dict[str, Any]) -> torch.Tensor:
    if "img_tv_transformed" in sample:
        return sample["img_tv_transformed"]
    return sample["img"]


def _result_row(path: str, names: list[str], probs, pred, target) -> dict[str, Any]:
    prob_v = np.asarray(probs, dtype=np.float64).reshape(-1)
    pred_b = _as_numpy_binary(pred)
    target_b = None if target is None else _as_numpy_binary(target)
    row: dict[str, Any] = {"img_path": path}
    for index, name in enumerate(names):
        row[f"prob_{name}"] = round(float(prob_v[index]), 6)
        row[f"pred_{name}"] = int(pred_b[index])
        if target_b is not None:
            row[f"gt_{name}"] = int(target_b[index])
    return row


def _write_metrics_csv(path: Path, names, accuracy, recall, counts: AttributeCount) -> None:
    assert counts.tp is not None and counts.gt_pos is not None
    rows = []
    for index, name in enumerate(names):
        recall_value = float(recall[index])
        rows.append(
            {
                "attribute": attribute_display_name(name),
                "accuracy": float(accuracy[index]),
                "recall": None if np.isnan(recall_value) else recall_value,
                "tp": int(counts.tp[index]),
                "gt_positive": int(counts.gt_pos[index]),
                "count": counts.count,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _panel_grid(panels: list[np.ndarray], nrow: int) -> np.ndarray:
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    tensors = []
    for panel in panels:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        tensors.append(BaseDataset.convert_img_from_numpy_to_tensor_uint8(canvas).float() / 255.0)
    grid = make_grid(tensors, nrow=nrow)
    return (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _add_image(trainer, pl_module, tag: str, image_rgb: np.ndarray) -> None:
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
    step = int(getattr(pl_module, "global_step", 0))
    for experiment in _experiments(trainer, pl_module):
        experiment.add_image(tag, tensor, global_step=step)


def _experiments(trainer, pl_module):
    loggers = list(getattr(trainer, "loggers", None) or [])
    if not loggers:
        logger = getattr(trainer, "logger", None) or getattr(pl_module, "logger", None)
        if logger is not None:
            loggers = [logger]
    for logger in loggers:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_image"):
            yield experiment


def _val_dataset(trainer):
    datamodule = getattr(trainer, "datamodule", None)
    return getattr(datamodule, "val_dataset", None) if datamodule is not None else None


def _loader_dataset(loader):
    if loader is None:
        return None
    if isinstance(loader, (list, tuple)):
        loader = loader[0] if loader else None
    return getattr(loader, "dataset", None)


def _is_rank_zero(trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


def _skip_rank_or_sanity(trainer) -> bool:
    return (not _is_rank_zero(trainer)) or bool(getattr(trainer, "sanity_checking", False))


def _as_binary_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    if value.dtype == torch.bool:
        return value
    return value >= 0.5


def _as_numpy_binary(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return _as_binary_tensor(value).numpy()
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    return array >= 0.5


def _swatch(image: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (x, y), (x + 8, y + 8), color, -1)


def _safe_name(name: str) -> str:
    text = str(name).strip() or "unknown"
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _jpg_path(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}.jpg"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = directory / f"{stem}_{index}.jpg"
        if not candidate.exists():
            return candidate
        index += 1
