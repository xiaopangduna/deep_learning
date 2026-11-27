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


def render_attribute_rate_chart(
    names: list[str],
    positive_rate: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
) -> np.ndarray:
    """正样本比例、精确率、召回率，稀有属性在上。返回 RGB uint8。没有正例或预测的位置留空。"""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [attribute_display_name(name) for name in names]
    positive = np.asarray(positive_rate, dtype=np.float64)
    precise = np.asarray(precision, dtype=np.float64)
    recalled = np.asarray(recall, dtype=np.float64)
    order = _order_by_ascending(positive)
    labels = [labels[index] for index in order]
    positive, precise, recalled = positive[order], precise[order], recalled[order]
    y = np.arange(len(labels))
    height = 0.25
    fig, axis = plt.subplots(figsize=(8.0, max(4.8, 0.32 * len(labels) + 1.2)))
    axis.barh(y - height, positive, height, label="positive rate", color="#B0B0B0")
    axis.barh(y, precise, height, label="precision", color="#4C78A8")
    axis.barh(y + height, recalled, height, label="recall", color="#F58518")
    axis.set_xlim(0.0, 1.0)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _rgb_from_matplotlib_figure(fig)


# 阈值 0.5 上能从混淆矩阵算出的率，再加上两条与阈值无关的排序曲线面积。
# 分母为 0 的位置是 NaN，例如该属性没有正样本时的召回率。
_RATE_COLUMNS = (
    "positive_rate",
    "predicted_positive_rate",
    "accuracy",
    "majority_accuracy",
    "lift",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "fnr",
    "npv",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
)
_COUNT_COLUMNS = (
    "tp",
    "fp",
    "tn",
    "fn",
    "gt_positive",
    "gt_negative",
    "pred_positive",
    "pred_negative",
    "count",
)
# 先看阳性率和准确率是否只是多数类，再看错在漏报还是误报，计数放最后。
_METRIC_DISPLAY_COLUMNS = (
    "positive_rate",
    "gt_positive",
    "count",
    "accuracy",
    "majority_accuracy",
    "lift",
    "precision",
    "recall",
    "f1",
    "specificity",
    "npv",
    "fpr",
    "fnr",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "predicted_positive_rate",
    "pred_positive",
    "pred_negative",
    "tp",
    "fp",
    "tn",
    "fn",
    "gt_negative",
)


class AttributeCount:
    """逐属性累计混淆矩阵，并保留概率以便计算 ROC-AUC 和 PR-AUC。"""

    def __init__(self) -> None:
        self.correct: torch.Tensor | None = None
        self.tp: torch.Tensor | None = None
        self.fp: torch.Tensor | None = None
        self.gt_pos: torch.Tensor | None = None
        self.count = 0
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, probs: torch.Tensor | None = None) -> None:
        pred_b = _as_binary_tensor(pred)
        target_b = _as_binary_tensor(target)
        if pred_b.shape != target_b.shape:
            raise ValueError(f"预测 {tuple(pred_b.shape)} 与真值 {tuple(target_b.shape)} 形状不一致")
        correct = pred_b.eq(target_b).sum(dim=0).to(torch.int64)
        tp = (pred_b & target_b).sum(dim=0).to(torch.int64)
        fp = (pred_b & ~target_b).sum(dim=0).to(torch.int64)
        positive = target_b.sum(dim=0).to(torch.int64)
        if self.correct is None:
            self.correct = correct
            self.tp = tp
            self.fp = fp
            self.gt_pos = positive
        elif self.correct.shape != correct.shape:
            raise ValueError(
                f"属性数从 {self.correct.shape[0]} 变成 {correct.shape[0]}"
            )
        else:
            self.correct += correct
            self.tp += tp
            self.fp += fp
            self.gt_pos += positive
        self.count += int(pred_b.shape[0])
        if probs is None:
            return
        scores = probs.detach().cpu().float()
        if scores.shape != target_b.shape:
            raise ValueError(f"概率 {tuple(scores.shape)} 与真值 {tuple(target_b.shape)} 形状不一致")
        self._scores.append(scores)
        self._labels.append(target_b.to(torch.int64))

    def accuracy_recall(self) -> tuple[np.ndarray, np.ndarray]:
        summary = self.summary()
        return summary["accuracy"], summary["recall"]

    def summary(self) -> dict[str, np.ndarray]:
        """每个属性一行的计数和比率。训练用的平均准确率不在这里改。"""
        if self.correct is None or self.tp is None or self.fp is None or self.gt_pos is None or self.count == 0:
            raise RuntimeError("还没有样本")
        tp = self.tp.numpy().astype(np.int64)
        fp = self.fp.numpy().astype(np.int64)
        gt_pos = self.gt_pos.numpy().astype(np.int64)
        count = np.full(tp.shape, self.count, dtype=np.int64)
        fn = gt_pos - tp
        gt_neg = count - gt_pos
        tn = gt_neg - fp
        pred_pos = tp + fp
        pred_neg = tn + fn
        positive_rate = _divide(gt_pos, count)
        predicted_positive_rate = _divide(pred_pos, count)
        accuracy = _divide(tp + tn, count)
        majority = np.maximum(positive_rate, 1.0 - positive_rate)
        precision = _divide(tp, pred_pos)
        recall = _divide(tp, gt_pos)
        specificity = _divide(tn, gt_neg)
        roc_auc, pr_auc = self._ranking(tp.shape[0])
        return {
            "positive_rate": positive_rate,
            "predicted_positive_rate": predicted_positive_rate,
            "accuracy": accuracy,
            "majority_accuracy": majority,
            "lift": accuracy - majority,
            "balanced_accuracy": _pair_mean(recall, specificity),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fpr": _divide(fp, gt_neg),
            "fnr": _divide(fn, gt_pos),
            "npv": _divide(tn, pred_neg),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
            "mcc": _divide(
                tp.astype(np.float64) * tn - fp.astype(np.float64) * fn,
                np.sqrt(
                    (tp + fp).astype(np.float64)
                    * (tp + fn).astype(np.float64)
                    * (tn + fp).astype(np.float64)
                    * (tn + fn).astype(np.float64)
                ),
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "gt_positive": gt_pos,
            "gt_negative": gt_neg,
            "pred_positive": pred_pos,
            "pred_negative": pred_neg,
            "count": count,
        }

    def _ranking(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        roc_auc = np.full(width, np.nan, dtype=np.float64)
        pr_auc = np.full(width, np.nan, dtype=np.float64)
        if not self._scores:
            return roc_auc, pr_auc
        from torchmetrics.functional.classification import binary_auroc, binary_average_precision

        scores = torch.cat(self._scores, dim=0)
        labels = torch.cat(self._labels, dim=0)
        for index in range(width):
            label = labels[:, index]
            if int(label.min()) == int(label.max()):
                continue
            score = scores[:, index]
            roc_auc[index] = float(binary_auroc(score, label))
            pr_auc[index] = float(binary_average_precision(score, label))
        return roc_auc, pr_auc


def _rgb_from_matplotlib_figure(fig) -> np.ndarray:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("统计图编码失败")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def attribute_distribution_frame(
    split_csv_paths: dict[str, list[str] | tuple[str, ...]],
    attribute_columns: list[str] | None = None,
) -> pd.DataFrame:
    """训练、验证、测试的属性阳性人数和阳性率。

    只读 CSV 标签，不读图像。``attribute_columns`` 为 ``None`` 时用训练表里的全部 ``attr_`` 列。
    差是该划分阳性率减去训练阳性率。
    """
    from ..dataset.attribute_classifier import AttributeFromDetectBoxDataset
    from ..dataset.base import BaseDataset

    splits = ("train", "val", "test")
    missing = [name for name in splits if not split_csv_paths.get(name)]
    if missing:
        raise ValueError(f"属性分布需要训练、验证、测试 CSV，缺少 {missing}")
    train_paths = [Path(path) for path in split_csv_paths["train"]]
    columns = AttributeFromDetectBoxDataset.resolve_attribute_columns(
        BaseDataset.read_csv_fieldnames(train_paths[0]),
        attribute_columns,
    )
    counts = {split: _split_positive_counts(split_csv_paths[split], columns) for split in splits}
    rows = []
    for index, column in enumerate(columns):
        row: dict[str, Any] = {"attribute": attribute_display_name(column)}
        train_positive, train_count = counts["train"][index], counts["train"][-1]
        train_rate = train_positive / train_count
        for split in splits:
            positive, count = counts[split][index], counts[split][-1]
            rate = positive / count
            row[f"{split}_positive"] = positive
            row[f"{split}_count"] = count
            row[f"{split}_rate"] = rate
        row["val_minus_train"] = row["val_rate"] - train_rate
        row["test_minus_train"] = row["test_rate"] - train_rate
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("train_rate", "val_rate", "test_rate", "val_minus_train", "test_minus_train"):
        frame[column] = frame[column].map(lambda value: round(float(value), 4))
    frame = frame.sort_values(["train_rate", "attribute"], ascending=True, kind="mergesort")
    return frame[
        [
            "attribute",
            "train_rate",
            "val_rate",
            "test_rate",
            "val_minus_train",
            "test_minus_train",
            "train_positive",
            "val_positive",
            "test_positive",
            "train_count",
            "val_count",
            "test_count",
        ]
    ].reset_index(drop=True)


def render_attribute_distribution_chart(frame: pd.DataFrame) -> np.ndarray:
    """每个属性三根横柱：训练、验证、测试的阳性率，稀有属性在上。返回 RGB uint8。"""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(name) for name in frame["attribute"]]
    y = np.arange(len(labels))
    height = 0.25
    fig, axis = plt.subplots(figsize=(8.0, max(4.8, 0.32 * len(labels) + 1.2)))
    axis.barh(y - height, frame["train_rate"], height, label="train", color="#4C78A8")
    axis.barh(y, frame["val_rate"], height, label="val", color="#F58518")
    axis.barh(y + height, frame["test_rate"], height, label="test", color="#54A24B")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("positive rate")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _rgb_from_matplotlib_figure(fig)


def _split_positive_counts(csv_paths, columns: list[str]) -> tuple[int, ...]:
    frames = []
    for path in csv_paths:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"属性分布 CSV 不存在：{file}")
        frames.append(pd.read_csv(file, dtype=str, keep_default_na=False, usecols=columns))
    table = pd.concat(frames, ignore_index=True)
    if len(table) == 0:
        raise ValueError(f"属性分布 CSV 没有数据行：{list(csv_paths)}")
    positives = []
    for column in columns:
        text = table[column].astype(str).str.strip()
        bad = ~text.isin({"0", "1"})
        if bool(bad.any()):
            shown = sorted(set(text[bad]))[:8]
            raise ValueError(f"属性列 {column} 只能是 0 或 1，收到 {shown}")
        positives.append(int((text == "1").sum()))
    positives.append(len(table))
    return tuple(positives)


class SaveAttributeDistributionCallback(pl.Callback):
    """把三个划分的属性分布写到本次日志目录。

    ``fit``、单独的 ``validate`` 和 ``test`` 各写一次。训练过程中每个 epoch 的验证
    不再重写，因为标签表不变，``on_fit_start`` 已经写过。
    文件是 ``attribute_distribution.csv`` 和 ``attribute_distribution.png``。
    ``save_dir`` 为空时用 ``trainer.log_dir``。
    """

    def __init__(self, save_dir: str | None = None) -> None:
        super().__init__()
        self._configured_save_dir = Path(save_dir) if save_dir else None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._write(trainer, pl_module, "fit/attribute_distribution")

    def on_validation_start(self, trainer, pl_module) -> None:
        if getattr(trainer, "training", False) or getattr(trainer, "sanity_checking", False):
            return
        self._write(trainer, pl_module, "val/attribute_distribution")

    def on_test_start(self, trainer, pl_module) -> None:
        self._write(trainer, pl_module, "test/attribute_distribution")

    def _write(self, trainer, pl_module, tag: str) -> None:
        if not _is_rank_zero(trainer):
            return
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "train_csv_paths"):
            return
        root = self._configured_save_dir
        if root is None:
            log_dir = getattr(trainer, "log_dir", None)
            root = Path(log_dir) if log_dir else None
        if root is None:
            print("Warning: attribute distribution skipped, no log_dir")
            return
        frame = attribute_distribution_frame(
            {
                "train": list(datamodule.train_csv_paths),
                "val": list(datamodule.val_csv_paths),
                "test": list(datamodule.test_csv_paths),
            },
            getattr(datamodule, "attribute_columns", None),
        )
        root.mkdir(parents=True, exist_ok=True)
        csv_path = root / "attribute_distribution.csv"
        frame.to_csv(csv_path, index=False)
        chart = render_attribute_distribution_chart(frame)
        png_path = root / "attribute_distribution.png"
        cv2.imwrite(str(png_path), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR))
        _add_image(trainer, pl_module, tag, chart)
        print(f"属性分布已保存到: {csv_path}")


class LogAttributeClassifierValVisualizationCallback(pl.Callback):
    """每个验证 epoch 把前几张脸写入 TensorBoard，并把逐属性分析指标追加到日志目录。

    脸来自该 epoch 第一个 batch 的前 ``max_images`` 张。验证集不打乱时，
    这就是 CSV 里的前几行，方便逐 epoch 看同一批脸。条形图和指标表用本进程
    看到的全部验证 batch。单卡时这是完整验证集。指标追加到
    ``val_per_attribute_metrics.csv``，宏平均追加到 ``val_metric_summary.csv``。
    训练日志里的 ``val_acc`` 仍是各属性准确率的平均。
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
            self._counts.update(pred, target, probs)
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
            rates = self._counts.summary()
            chart = render_attribute_rate_chart(
                names, rates["positive_rate"], rates["precision"], rates["recall"]
            )
            _add_image(trainer, pl_module, "val/attribute_rates", chart)
            _add_metric_means(trainer, pl_module, "val/metric_mean", rates)
            root = _log_dir(trainer)
            if root is None:
                return
            epoch = int(getattr(trainer, "current_epoch", 0))
            _append_csv(root / "val_per_attribute_metrics.csv", metrics_frame(names, rates, epoch))
            _append_csv(root / "val_metric_summary.csv", metric_summary_frame(rates, epoch))
            rank = detection_rank_frame(names, rates, epoch)
            rank_csv = root / "val_detection_rank.csv"
            rank.to_csv(rank_csv, index=False)
            rank_chart = render_detection_rank_chart(rank)
            cv2.imwrite(str(root / "val_detection_rank.png"), cv2.cvtColor(rank_chart, cv2.COLOR_RGB2BGR))
            _add_image(trainer, pl_module, "val/detection_rank", rank_chart)
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
        {root}/test_metric_summary.csv
        {root}/test_detection_rank.csv
        {root}/test_detection_rank.png
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
        self._counts.update(pred, target, probs)
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
        rates = self._counts.summary()
        metrics_path = root / "per_attribute_metrics.csv"
        metrics_frame(names, rates).to_csv(metrics_path, index=False)
        summary_path = root / "test_metric_summary.csv"
        metric_summary_frame(rates).to_csv(summary_path, index=False)
        chart = render_attribute_rate_chart(
            names, rates["positive_rate"], rates["precision"], rates["recall"]
        )
        chart_path = root / "per_attribute_metrics.png"
        cv2.imwrite(str(chart_path), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR))
        _add_image(trainer, pl_module, "test/attribute_rates", chart)
        rank = detection_rank_frame(names, rates)
        rank_path = root / "test_detection_rank.csv"
        rank.to_csv(rank_path, index=False)
        rank_chart = render_detection_rank_chart(rank)
        cv2.imwrite(str(root / "test_detection_rank.png"), cv2.cvtColor(rank_chart, cv2.COLOR_RGB2BGR))
        _add_image(trainer, pl_module, "test/detection_rank", rank_chart)
        _add_metric_means(trainer, pl_module, "test/metric_mean", rates)
        print(f"逐属性指标已保存到: {metrics_path}")
        print(f"按 F1 排序的检测效果已保存到: {rank_path}")

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


def _divide(num, den) -> np.ndarray:
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=np.float64)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _pair_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full(left.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    out[mask] = 0.5 * (left[mask] + right[mask])
    return out


def metrics_frame(names, rates: dict[str, np.ndarray], epoch: int | None = None) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(names):
        row: dict[str, Any] = {"attribute": attribute_display_name(name)}
        if epoch is not None:
            row["epoch"] = int(epoch)
        for key in _RATE_COLUMNS:
            value = float(rates[key][index])
            row[key] = None if np.isnan(value) else round(value, 4)
        for key in _COUNT_COLUMNS:
            row[key] = int(rates[key][index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["positive_rate", "attribute"], ascending=True, na_position="last", kind="mergesort")
    columns = ["attribute"] + (["epoch"] if epoch is not None else []) + list(_METRIC_DISPLAY_COLUMNS)
    return frame[columns].reset_index(drop=True)


def metric_summary_frame(rates: dict[str, np.ndarray], epoch: int | None = None) -> pd.DataFrame:
    """40 个属性的宏平均。某个属性分母为 0 时不进入该项平均。计数是各属性相加。"""
    row: dict[str, Any] = {}
    if epoch is not None:
        row["epoch"] = int(epoch)
    for key in _RATE_COLUMNS:
        values = np.asarray(rates[key], dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[key] = None if finite.size == 0 else round(float(finite.mean()), 4)
    for key in _COUNT_COLUMNS:
        if key == "count":
            row[key] = int(rates[key][0])
        else:
            row[key] = int(np.asarray(rates[key]).sum())
    columns = (["epoch"] if epoch is not None else []) + list(_METRIC_DISPLAY_COLUMNS)
    return pd.DataFrame([row])[columns]


def detection_rank_frame(names, rates: dict[str, np.ndarray], epoch: int | None = None) -> pd.DataFrame:
    """按 F1 从高到低排列。F1 同时看准和看全，用来比哪个属性检测效果好。"""
    rows = []
    for index, name in enumerate(names):
        row: dict[str, Any] = {"attribute": attribute_display_name(name)}
        for key in ("f1", "recall", "accuracy"):
            value = float(rates[key][index])
            row[key] = None if np.isnan(value) else round(value, 4)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["f1", "attribute"], ascending=[False, True], na_position="last", kind="mergesort")
    frame.insert(0, "rank", range(1, len(frame) + 1))
    if epoch is not None:
        frame.insert(0, "epoch", int(epoch))
    return frame.reset_index(drop=True)


def render_detection_rank_chart(frame: pd.DataFrame) -> np.ndarray:
    """F1 高的在上。F1 一列按高低着色。返回 RGB uint8。"""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    show = frame.loc[:, ["rank", "attribute", "f1", "recall", "accuracy"]]
    fig, axis = plt.subplots(figsize=(7.4, max(4.2, 0.28 * len(show) + 0.9)))
    axis.axis("off")
    axis.set_title("detection rank by F1", loc="left", fontsize=11, pad=8)
    cells = []
    for _, row in show.iterrows():
        cells.append(
            [
                str(int(row["rank"])),
                str(row["attribute"]),
                _rank_cell(row["f1"]),
                _rank_cell(row["recall"]),
                _rank_cell(row["accuracy"]),
            ]
        )
    table = axis.table(
        cellText=cells,
        colLabels=["rank", "attribute", "f1", "recall", "accuracy"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    f1_values = pd.to_numeric(show["f1"], errors="coerce").to_numpy(dtype=np.float64)
    finite = f1_values[np.isfinite(f1_values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    span = max(high - low, 1e-6)
    for row_index, value in enumerate(f1_values, start=1):
        if not np.isfinite(value):
            continue
        shade = (float(value) - low) / span
        table[row_index, 2].set_facecolor((1.0 - 0.55 * shade, 0.93 - 0.25 * shade, 0.75 - 0.55 * shade))
    fig.tight_layout()
    return _rgb_from_matplotlib_figure(fig)


def _rank_cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.4f}"


def _order_by_ascending(values: np.ndarray) -> np.ndarray:
    """NaN 放在最后，其余按升序，相同值保持原顺序。"""
    ranked = np.where(np.isfinite(values), values, np.inf)
    return np.argsort(ranked, kind="mergesort")


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _log_dir(trainer) -> Path | None:
    log_dir = getattr(trainer, "log_dir", None)
    return Path(log_dir) if log_dir else None


def _add_metric_means(trainer, pl_module, tag: str, rates: dict[str, np.ndarray]) -> None:
    step = int(getattr(pl_module, "global_step", 0))
    for experiment in _experiments(trainer, pl_module, "add_scalar"):
        for key in _RATE_COLUMNS:
            values = np.asarray(rates[key], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            experiment.add_scalar(f"{tag}/{key}", float(finite.mean()), global_step=step)


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


def _experiments(trainer, pl_module, method: str = "add_image"):
    loggers = list(getattr(trainer, "loggers", None) or [])
    if not loggers:
        logger = getattr(trainer, "logger", None) or getattr(pl_module, "logger", None)
        if logger is not None:
            loggers = [logger]
    for logger in loggers:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, method):
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
