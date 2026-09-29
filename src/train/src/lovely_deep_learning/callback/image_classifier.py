"""图像分类 Lightning 回调。

- ``Log*TrainVal*``：train / val → TensorBoard 图像
- ``Save*TestPredict*``：test / predict → 本地目录（标注图 + CSV）

两类 Callback 均依赖 :class:`~lovely_deep_learning.module.base.BaseModule` 的 step 返回值
（``metric_preds`` / ``net_out``）及 collate 后的 ``net_in: tuple[dict]``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from ..dataset.image_classifier import ImageClassifierDataset


class LogImageClassifierTrainValVisualizationCallback(pl.Callback):
    """train / val 阶段将分类可视化写入 TensorBoard（不落盘）。

    **触发时机**

    - :meth:`on_train_batch_end` / :meth:`on_validation_batch_end`（val 仅 ``dataloader_idx==0``）
    - 仅 ``trainer.is_global_zero`` 且 ``batch_idx==0`` 时执行（每 epoch train/val 各记一次）
    - TensorBoard tag：``train/sample_classifications``、``val/sample_classifications``

    **调用链**

    ``on_*_batch_end`` → :meth:`_try_log_batch` → :meth:`_log_sample_classifications_tensorboard`

    **数据契约**

    Module（:class:`~lovely_deep_learning.module.image_classifier.ImageClassifierModule`，
    继承 :class:`~lovely_deep_learning.module.base.BaseModule`）：

    - ``training_step`` → ``{"loss", "metric_preds", "net_out"}``
    - ``validation_step`` → ``{"metric_preds", "net_out"}``
    - ``metric_preds["pred_ids"]`` / ``["pred_conf"]``：postprocess 的 batch 级预测
    - ``net_out["class_id"]`` / ``["class_name"]``：collate 后的 GT（batched dict）

    Batch（:class:`ImageClassifierDataset` collate）：``(net_in, net_out)``；
    ``net_in`` 为 ``tuple[dict]``，绘制时取 ``net_in[i]["img_tv_transformed"]``。

    Datamodule：``trainer.datamodule.train_dataset`` / ``val_dataset``（:class:`ImageClassifierDataset`），
    提供 ``map_class_id_to_class_name`` 及 tensor ↔ numpy 绘制工具。

    Logger：``pl_module.logger.experiment`` 须支持 ``add_image``（如 TensorBoardLogger）；
    无 logger 或不支持时静默跳过。

    **参数**

    ``max_images``（默认 4）、``nrow``（默认 2）：经 :func:`~torchvision.utils.make_grid` 拼成网格后写入。
    """

    def __init__(
        self,
        max_images: int = 4,
        nrow: int = 2,
    ) -> None:
        super().__init__()
        self.max_images = int(max_images)
        self.nrow = int(nrow)

    def _log_sample_classifications_tensorboard(
        self,
        pl_module: pl.LightningModule,
        batch: Any,
        outputs: dict[str, Any],
        dataset: ImageClassifierDataset,
        tb_tag: str,
    ) -> None:
        """绘制单 batch 样本面板并 ``experiment.add_image``。

        数据契约见 :class:`LogImageClassifierTrainValVisualizationCallback`。
        ``dataset`` 由 :meth:`_try_log_batch` 从 datamodule 注入，使用：

        - :meth:`~lovely_deep_learning.dataset.image_classifier.ImageClassifierDataset.convert_img_from_tensor_to_numpy`
        - :meth:`~lovely_deep_learning.dataset.image_classifier.ImageClassifierDataset.draw_target_and_predict_label_on_numpy`
        - :meth:`~lovely_deep_learning.dataset.base.BaseDataset.convert_img_from_numpy_to_tensor_uint8`
        """
        metric_preds = outputs.get("metric_preds")
        net_out = outputs.get("net_out")
        if metric_preds is None or net_out is None:
            return
        if getattr(pl_module, "logger", None) is None:
            return
        try:
            experiment = pl_module.logger.experiment
        except Exception:
            return
        if not hasattr(experiment, "add_image"):
            return

        net_in, _ = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        n = min(self.max_images, len(net_in))
        if n == 0:
            return

        panels: list[torch.Tensor] = []
        for i in range(n):
            class_name = net_out["class_name"][i]
            class_id = net_out["class_id"][i]
            class_id_pred = pred_ids[i].item()
            class_name_pred = dataset.map_class_id_to_class_name[class_id_pred]
            confidence_pred = pred_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(
                net_in[i]["img_tv_transformed"]
            )
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=class_name,
                class_id=class_id,
                class_name_pred=class_name_pred,
                class_id_pred=class_id_pred,
                class_id_conf=confidence_pred,
            )
            panels.append(dataset.convert_img_from_numpy_to_tensor_uint8(img_np))

        img_grid = make_grid(panels, nrow=self.nrow)
        experiment.add_image(
            tb_tag,
            img_grid,
            global_step=pl_module.global_step,
        )

    def _try_log_batch(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataset_name: str,
        tb_tag: str,
        phase_label: str,
    ) -> None:
        """rank0 / ``batch_idx==0`` 门禁；通过后调用 :meth:`_log_sample_classifications_tensorboard`。

        数据契约见 :class:`LogImageClassifierTrainValVisualizationCallback`。
        ``dataset_name`` 为 ``"train_dataset"`` 或 ``"val_dataset"``。
        """
        if not trainer.is_global_zero or batch_idx != 0:
            return
        if outputs is None or not isinstance(outputs, dict):
            return
        dm = trainer.datamodule
        if dm is None:
            return
        dataset = getattr(dm, dataset_name, None)
        if dataset is None:
            return
        try:
            self._log_sample_classifications_tensorboard(
                pl_module, batch, outputs, dataset, tb_tag
            )
        except Exception as e:
            print(
                f"Warning: failed to log {phase_label} classification visualization at step "
                f"{pl_module.global_step}, {e}"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._try_log_batch(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            "train_dataset",
            "train/sample_classifications",
            "train",
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0:
            return
        self._try_log_batch(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            "val_dataset",
            "val/sample_classifications",
            "val",
        )


def _path_component(name: str) -> str:
    text = str(name).strip() or "unknown"
    return text.replace("/", "_").replace("\\", "_")


def error_pair_dirname(class_name: str, class_name_pred: str) -> str:
    """错分目录名：``{真值类名}__{预测类名}``。"""
    return f"{_path_component(class_name)}__{_path_component(class_name_pred)}"


def error_pair_bgr(class_id: int, class_id_pred: int) -> tuple[int, int, int]:
    """同一 ``(真值 id, 预测 id)`` 始终得到同一 BGR 边框色。"""
    hue = int((int(class_id) * 47 + int(class_id_pred) * 97) * 17) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_error_pair_border(
    img: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 4,
) -> np.ndarray:
    h, w = img.shape[:2]
    t = max(1, min(int(thickness), max(1, h // 2), max(1, w // 2)))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, t)
    return img


def unique_jpg_path(directory: Path, stem: str) -> Path:
    """``stem.jpg``；已存在时用 ``stem_1.jpg``、``stem_2.jpg``。"""
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


def class_names_for_ids(
    mapping: dict[int, str],
    target_ids: list[int],
    pred_ids: list[int],
) -> list[str]:
    ids = [int(i) for i in [*target_ids, *pred_ids, *mapping.keys()]]
    if not ids:
        return []
    count = max(ids) + 1
    return [str(mapping.get(i, i)) for i in range(count)]


def write_confusion_matrix_png(
    target_ids: list[int],
    pred_ids: list[int],
    class_names: list[str],
    path: Path,
) -> None:
    """行是真值、列是预测，单元格为样本数。"""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchmetrics.classification import MulticlassConfusionMatrix

    num_classes = len(class_names)
    counts = (
        MulticlassConfusionMatrix(num_classes=num_classes)(
            torch.tensor(pred_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )
        .detach()
        .cpu()
        .to(torch.int64)
        .numpy()
    )
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.9 * num_classes + 2), max(5.0, 0.9 * num_classes + 2))
    )
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


class SaveImageClassifierTestPredictVisualizationCallback(pl.Callback):
    """test / predict 阶段将分类可视化写入本地。

    与 :class:`LogImageClassifierTrainValVisualizationCallback` 分工：后者仅 train/val → TensorBoard。

    **触发时机**

    - :meth:`on_test_batch_end`：累计真值 / 预测 id；错分样本按 ``{真值}__{预测}`` 写图
    - :meth:`on_test_end`：写混淆矩阵图，并 ``add_image`` 到 TensorBoard
    - :meth:`on_predict_batch_end`：无 GT，写预测图
    - ``save_dir`` 为 ``None`` 时使用 ``trainer.log_dir``（本次 test / predict 的日志目录）

    **数据契约**

    Module（:class:`~lovely_deep_learning.module.base.BaseModule`）：

    - ``test_step`` → ``{"metric_preds", "net_out"}``
    - ``predict_step`` → ``{"metric_preds"}``
    - ``metric_preds["pred_ids"]`` / ``["pred_conf"]``：batch 级张量

    Batch（:class:`ImageClassifierDataset` collate）：``(net_in, net_out)``；
    ``net_in[i]["img_path"]`` / ``["img_tv_transformed"]`` 用于读图与绘制；
    test 阶段 GT 来自 ``net_out["class_id"]`` / ``["class_name"]``。

    Dataloader dataset：``trainer.test_dataloaders.dataset`` /
    ``trainer.predict_dataloaders.dataset``（:class:`ImageClassifierDataset`），
    供类名映射与绘制工具。

    **目录结构**（根目录为 ``save_dir``，缺省为 ``trainer.log_dir``）::

        {root}/confusion_matrix.png
        {root}/test/{gt}__{pred}/*.jpg
        {root}/test_results.csv
        {root}/predict/*.jpg
        {root}/prediction_results.csv

    **参数**

    ``test_only_save_mistake=True`` 时，test 阶段仅保存预测错误样本图。
    错分图带按 ``(class_id, class_id_pred)`` 固定的边框色。
    """

    def __init__(self, save_dir: str | None = None, test_only_save_mistake: bool = True):
        """
        Args:
            save_dir: 写入根目录。为 ``None`` 时用本次运行的 ``trainer.log_dir``。
            test_only_save_mistake: 测试时是否仅保存预测错误样本图。
        """
        super().__init__()
        self.test_only_save_mistake = test_only_save_mistake
        self._configured_save_dir = Path(save_dir) if save_dir else None
        self.save_dir: Path | None = None
        self.save_dir_test: Path | None = None
        self.save_dir_pred: Path | None = None
        self._dirs_ready = False
        self.csv_table_test: list[dict[str, Any]] = []
        self.csv_table_pred: list[dict[str, Any]] = []
        self._test_target_ids: list[int] = []
        self._test_pred_ids: list[int] = []

    def _prepare_output_dirs(self, trainer) -> bool:
        if self._dirs_ready:
            return self.save_dir is not None
        root = self._configured_save_dir
        if root is None:
            root = getattr(trainer, "log_dir", None)
            if not root:
                logger = getattr(trainer, "logger", None)
                root = getattr(logger, "log_dir", None) if logger is not None else None
        if not root:
            self.save_dir = None
            self._dirs_ready = True
            return False
        self.save_dir = Path(root)
        self.save_dir_test = self.save_dir / "test"
        self.save_dir_pred = self.save_dir / "predict"
        self.save_dir_test.mkdir(parents=True, exist_ok=True)
        self.save_dir_pred.mkdir(parents=True, exist_ok=True)
        self._dirs_ready = True
        return True

    def _log_confusion_matrix(self, trainer, pl_module, image_rgb: np.ndarray) -> None:
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
        loggers = list(getattr(trainer, "loggers", None) or [])
        if not loggers and getattr(trainer, "logger", None) is not None:
            loggers = [trainer.logger]
        step = int(getattr(pl_module, "global_step", 0))
        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "add_image"):
                continue
            try:
                experiment.add_image("test/confusion_matrix", tensor, global_step=step)
            except Exception as exc:
                print(f"Warning: failed to log confusion matrix to TensorBoard: {exc}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """逐样本绘制 GT + 预测标签。错分样本写入 ``{真值}__{预测}/`` 并加上该错分对的边框。"""
        if outputs is None or not self._prepare_output_dirs(trainer):
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ImageClassifierDataset = trainer.test_dataloaders.dataset

        net_in, net_out = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        for i in range(len(net_in)):
            img_path = Path(net_in[i]["img_path"])
            img = net_in[i]["img_tv_transformed"]

            cur_class_id = int(net_out["class_id"][i].item())
            cur_class_name = net_out["class_name"][i]

            cur_class_id_pred = int(pred_ids[i].item())
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = pred_conf[i].item()
            is_correct = cur_class_id == cur_class_id_pred

            self._test_target_ids.append(cur_class_id)
            self._test_pred_ids.append(cur_class_id_pred)

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=cur_class_name,
                class_id=cur_class_id,
                class_name_pred=cur_class_name_pred,
                class_id_pred=cur_class_id_pred,
                class_id_conf=cur_confidence_pred,
            )
            if not is_correct:
                img_np = draw_error_pair_border(
                    img_np,
                    error_pair_bgr(cur_class_id, cur_class_id_pred),
                )

            save_path = ""
            if not (self.test_only_save_mistake and is_correct):
                pair_dir = self.save_dir_test / error_pair_dirname(
                    cur_class_name, cur_class_name_pred
                )
                save_path = str(unique_jpg_path(pair_dir, img_path.stem))
                cv2.imwrite(save_path, img_np)

            self.csv_table_test.append(
                {
                    "img_path": str(img_path),
                    "class_id": cur_class_id,
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "class_name": cur_class_name,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": save_path,
                }
            )

    def on_test_epoch_start(self, trainer, pl_module):
        self.csv_table_test = []
        self._test_target_ids = []
        self._test_pred_ids = []
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        if self._test_target_ids and self._prepare_output_dirs(trainer):
            dataset: ImageClassifierDataset = trainer.test_dataloaders.dataset
            class_names = class_names_for_ids(
                dataset.map_class_id_to_class_name,
                self._test_target_ids,
                self._test_pred_ids,
            )
            matrix_path = self.save_dir / "confusion_matrix.png"
            write_confusion_matrix_png(
                self._test_target_ids,
                self._test_pred_ids,
                class_names,
                matrix_path,
            )
            matrix_bgr = cv2.imread(str(matrix_path), cv2.IMREAD_COLOR)
            if matrix_bgr is not None:
                self._log_confusion_matrix(
                    trainer,
                    pl_module,
                    cv2.cvtColor(matrix_bgr, cv2.COLOR_BGR2RGB),
                )
            print(f"混淆矩阵已保存到: {matrix_path}")
            print(f"测试图片已保存到: {self.save_dir_test}")
        if self.csv_table_test and self.save_dir is not None:
            df = pd.DataFrame(self.csv_table_test)
            csv_save_path = self.save_dir / "test_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """预测阶段无 GT，仅绘制预测标签并写入 ``predict/``。"""
        if outputs is None or not self._prepare_output_dirs(trainer):
            return
        metric_preds = outputs.get("metric_preds")
        if metric_preds is None:
            return
        dataset: ImageClassifierDataset = trainer.predict_dataloaders.dataset

        net_in, _net_out = batch
        pred_ids = metric_preds["pred_ids"]
        pred_conf = metric_preds["pred_conf"]

        B = len(net_in)

        for i in range(B):
            img_path = Path(net_in[i]["img_path"])
            img = net_in[i]["img_tv_transformed"]
            cur_class_id_pred = pred_ids[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = pred_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np, class_name_pred=cur_class_name_pred, class_id_pred=cur_class_id_pred, class_id_conf=cur_confidence_pred
            )

            save_path = unique_jpg_path(self.save_dir_pred, img_path.stem)
            cv2.imwrite(str(save_path), img_np)

            self.csv_table_pred.append(
                {
                    "img_path": str(img_path),
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": str(save_path),
                }
            )

    def on_predict_epoch_start(self, trainer, pl_module):
        self.csv_table_pred = []
        return super().on_predict_epoch_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.csv_table_pred:
            df = pd.DataFrame(self.csv_table_pred)
            csv_save_path = self.save_dir / "prediction_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")
