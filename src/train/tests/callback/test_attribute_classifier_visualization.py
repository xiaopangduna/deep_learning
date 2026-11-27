"""属性可视化：只画阳性属性，测试阶段按误报 / 漏报封顶落盘。"""

from types import SimpleNamespace

import cv2
import pytest
import numpy as np
import pandas as pd
import torch

from lovely_deep_learning.callback.attribute_classifier import (
    COLOR_FP,
    LogAttributeClassifierValVisualizationCallback,
    SaveAttributeClassifierTestPredictVisualizationCallback,
    SaveAttributeDistributionCallback,
    _face_bgr,
    attribute_distribution_frame,
    choose_f1_thresholds,
    render_attribute_panel,
    visible_attribute_rows,
)


class _Dataset:
    attribute_columns = ["attr_Male", "attr_Smiling"]
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]


class _Experiment:
    def __init__(self):
        self.images = []

    def add_image(self, tag, img, global_step=0):
        self.images.append((tag, tuple(img.shape), global_step))


def _trainer(tmp_path, loader_name=None):
    logger = SimpleNamespace(experiment=_Experiment(), log_dir=str(tmp_path))
    trainer = SimpleNamespace(
        log_dir=str(tmp_path),
        logger=logger,
        loggers=[logger],
        is_global_zero=True,
        sanity_checking=False,
        datamodule=SimpleNamespace(val_dataset=_Dataset()),
    )
    if loader_name is not None:
        setattr(trainer, loader_name, SimpleNamespace(dataset=_Dataset()))
    return trainer, logger.experiment


def _batch(path, target, pred, probs):
    net_in = ({"img_path": path, "img_tv_transformed": torch.zeros(3, 16, 16)},)
    net_out = {} if target is None else {"attr": torch.tensor([target], dtype=torch.float32)}
    outputs = {
        "metric_preds": {
            "pred": torch.tensor([pred], dtype=torch.float32),
            "probs": torch.tensor([probs], dtype=torch.float32),
        }
    }
    return outputs, (net_in, net_out)


def test_visible_rows_drop_true_negatives_and_sort_errors_first():
    rows = visible_attribute_rows(
        ["A", "B", "C", "D"],
        probs=[0.2, 0.95, 0.7, 0.1],
        pred=[0, 1, 1, 0],
        target=[1, 0, 1, 0],
    )
    assert [item[0] for item in rows] == ["B", "A", "C"]
    assert [item[2] for item in rows] == ["fp", "fn", "tp"]


def test_panel_marks_false_positive_with_solid_color():
    face = np.zeros((24, 24, 3), dtype=np.uint8)
    panel = render_attribute_panel(face, [("Smiling", 0.91, "fp")], with_target=True)
    assert panel.shape[1] > face.shape[1]
    right = panel[:, face.shape[1] + 4 :]
    assert np.all(right == np.array(COLOR_FP), axis=2).any()


def test_denormalize_uses_dataset_mean_and_std():
    # -0.5 落在标准化区间外，乘标准差加均值后是 0.25。
    face = _face_bgr(_Dataset(), {"img_tv_transformed": torch.full((3, 8, 8), -0.5)})
    assert face.shape == (8, 8, 3)
    assert 60 <= int(face.mean()) <= 70


def test_val_logs_first_batch_and_metric_chart(tmp_path, monkeypatch):
    calls = []

    def fake_face(dataset, sample):
        calls.append(sample["img_path"])
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "lovely_deep_learning.callback.attribute_classifier._face_bgr",
        fake_face,
    )
    trainer, experiment = _trainer(tmp_path)
    module = SimpleNamespace(global_step=4, logger=trainer.logger)
    callback = LogAttributeClassifierValVisualizationCallback(max_images=2, nrow=2)
    callback.on_validation_epoch_start(trainer, module)

    batch0_in = tuple(
        {"img_path": f"/data/{index}.jpg", "img_tv_transformed": torch.zeros(3, 8, 8)}
        for index in range(3)
    )
    batch0_out = {"attr": torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)}
    callback.on_validation_batch_end(
        trainer,
        module,
        {
            "metric_preds": {
                "pred": torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32),
                "probs": torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.4]], dtype=torch.float32),
            }
        },
        (batch0_in, batch0_out),
        0,
    )
    later_outputs, later_batch = _batch("/data/later.jpg", [1, 0], [1, 0], [0.6, 0.2])
    callback.on_validation_batch_end(trainer, module, later_outputs, later_batch, 1)
    callback.on_validation_epoch_end(trainer, module)

    assert calls == ["/data/0.jpg", "/data/1.jpg"]
    tags = [item[0] for item in experiment.images]
    assert tags == ["val/sample_attributes", "val/attribute_rates", "val/detection_rank"]
    assert experiment.images[0][2] == 4
    table = pd.read_csv(tmp_path / "val_per_attribute_metrics.csv")
    assert list(table.columns[:6]) == ["attribute", "ap", "threshold", "f1", "precision", "recall"]
    assert (tmp_path / "val_metric_summary.csv").is_file()
    rank = pd.read_csv(tmp_path / "val_detection_rank.csv")
    assert list(rank.columns) == [
        "rank",
        "attribute",
        "ap",
        "positive_rate",
        "ap_minus_positive_rate",
        "threshold",
        "f1",
        "precision",
        "recall",
    ]
    assert (tmp_path / "val_detection_rank.png").is_file()


def test_val_second_batch_call_uses_outputs(tmp_path, monkeypatch):
    """第二个 batch 只累计指标，不再画脸。"""
    monkeypatch.setattr(
        "lovely_deep_learning.callback.attribute_classifier._face_bgr",
        lambda dataset, sample: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    trainer, experiment = _trainer(tmp_path)
    module = SimpleNamespace(global_step=1, logger=trainer.logger)
    callback = LogAttributeClassifierValVisualizationCallback(max_images=8)
    outputs, batch = _batch("/data/a.jpg", [1, 0], [0, 0], [0.2, 0.1])
    callback.on_validation_batch_end(trainer, module, outputs, batch, 1)
    callback.on_validation_epoch_end(trainer, module)
    assert [item[0] for item in experiment.images] == ["val/attribute_rates", "val/detection_rank"]
    assert callback._counts.count == 1
    accuracy, recall = callback._counts.accuracy_recall()
    assert accuracy[0] == 0
    assert recall[0] == 0


def test_test_end_caps_error_images_and_writes_metrics(tmp_path):
    trainer, experiment = _trainer(tmp_path, "test_dataloaders")
    module = SimpleNamespace(global_step=2, logger=trainer.logger)
    callback = SaveAttributeClassifierTestPredictVisualizationCallback(
        save_dir=str(tmp_path),
        max_per_bucket=2,
    )
    callback._thresholds = np.array([0.5, 0.5])
    samples = [
        ("/data/a.jpg", [1, 0], [1, 1], [0.9, 0.8]),
        ("/data/b.jpg", [1, 1], [0, 1], [0.2, 0.7]),
        ("/data/c.jpg", [0, 0], [0, 0], [0.1, 0.1]),
        ("/data/d.jpg", [0, 0], [0, 1], [0.1, 0.6]),
        ("/data/e.jpg", [0, 0], [0, 1], [0.1, 0.55]),
    ]
    for index, sample in enumerate(samples):
        outputs, batch = _batch(*sample)
        callback.on_test_batch_end(trainer, module, outputs, batch, index)
    callback.on_test_end(trainer, module)

    smiling_fp = list((tmp_path / "test" / "Smiling" / "fp").glob("*.jpg"))
    assert len(smiling_fp) == 2
    assert (tmp_path / "test" / "Male" / "fn" / "b.jpg").is_file()
    assert not (tmp_path / "test" / "Male" / "fp").exists()

    table = pd.read_csv(tmp_path / "test_attribute_probs.csv")
    assert list(table["img_path"]) == [item[0] for item in samples]
    assert table.loc[0, "prob_Male"] == 0.9
    assert table.loc[0, "gt_Smiling"] == 0
    assert "gt_Male" in table.columns

    metrics = pd.read_csv(tmp_path / "per_attribute_metrics.csv").set_index("attribute")
    assert metrics.loc["Male", "tp"] == 1
    assert metrics.loc["Male", "tn"] == 3
    assert metrics.loc["Male", "fn"] == 1
    assert metrics.loc["Male", "gt_positive"] == 2
    assert list(metrics.index) == ["Male", "Smiling"]
    assert metrics.loc["Male", "threshold"] == 0.5
    assert metrics.loc["Male", "ap"] == 1
    assert metrics.loc["Male", "ap_minus_positive_rate"] == pytest.approx(0.6, abs=5e-4)
    assert metrics.loc["Male", "recall"] == 0.5
    assert metrics.loc["Male", "precision"] == 1
    assert metrics.loc["Male", "f1"] == pytest.approx(2 / 3, abs=5e-4)
    assert metrics.loc["Male", "balanced_accuracy"] == 0.75
    assert metrics.loc["Smiling", "fp"] == 3
    assert metrics.loc["Smiling", "recall"] == 1
    assert metrics.loc["Smiling", "precision"] == 0.25
    assert metrics.loc["Smiling", "ap"] == 0.5
    summary = pd.read_csv(tmp_path / "test_metric_summary.csv")
    assert summary.loc[0, "count"] == 5
    assert summary.loc[0, "tp"] == 2
    rank = pd.read_csv(tmp_path / "test_detection_rank.csv")
    assert list(rank["attribute"]) == ["Male", "Smiling"]
    assert list(rank["rank"]) == [1, 2]
    assert list(rank.columns) == [
        "rank",
        "attribute",
        "ap",
        "positive_rate",
        "ap_minus_positive_rate",
        "threshold",
        "f1",
        "precision",
        "recall",
    ]
    assert (tmp_path / "test_detection_rank.png").is_file()
    assert (tmp_path / "per_attribute_metrics.png").is_file()
    assert cv2.imread(str(tmp_path / "per_attribute_metrics.png")) is not None
    assert experiment.images[0][0] == "test/attribute_rates"


def test_predict_draws_only_positive_predictions(tmp_path):
    trainer, _experiment = _trainer(tmp_path, "predict_dataloaders")
    module = SimpleNamespace(global_step=0, logger=trainer.logger)
    callback = SaveAttributeClassifierTestPredictVisualizationCallback(
        save_dir=str(tmp_path),
        max_predict_images=1,
    )
    first, first_batch = _batch("/data/one.jpg", None, [1, 0], [0.81, 0.2])
    second, second_batch = _batch("/data/two.jpg", None, [0, 1], [0.1, 0.9])
    callback.on_predict_batch_end(trainer, module, first, first_batch, 0)
    callback.on_predict_batch_end(trainer, module, second, second_batch, 1)
    callback.on_predict_end(trainer, module)

    saved = list((tmp_path / "predict").glob("*.jpg"))
    assert [path.name for path in saved] == ["one.jpg"]
    table = pd.read_csv(tmp_path / "prediction_attribute_probs.csv")
    assert list(table["img_path"]) == ["/data/one.jpg", "/data/two.jpg"]
    assert table.loc[0, "pred_Male"] == 1
    assert table.loc[0, "pred_Smiling"] == 0
    assert "gt_Male" not in table.columns


def test_predict_with_labels_writes_ground_truth(tmp_path):
    trainer, _experiment = _trainer(tmp_path, "predict_dataloaders")
    module = SimpleNamespace(global_step=0, logger=trainer.logger)
    callback = SaveAttributeClassifierTestPredictVisualizationCallback(save_dir=str(tmp_path))
    outputs, batch = _batch("/data/one.jpg", [1, 0], [1, 1], [0.9, 0.8])
    callback.on_predict_batch_end(trainer, module, outputs, batch, 0)
    callback.on_predict_end(trainer, module)
    table = pd.read_csv(tmp_path / "prediction_attribute_probs.csv")
    assert table.loc[0, "gt_Male"] == 1
    assert table.loc[0, "gt_Smiling"] == 0
    assert (tmp_path / "predict" / "one.jpg").is_file()


def _write_attr_csv(path, rows):
    path.write_text(
        "path_img,attr_Bald,attr_Male\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_choose_f1_threshold_keeps_the_cut_closest_to_half():
    scores = np.array([[0.3, 0.9], [0.05, 0.2], [0.0, 0.1]])
    labels = np.array([[1, 1], [0, 0], [0, 0]])
    thresholds = choose_f1_thresholds(scores, labels)
    assert thresholds[0] == pytest.approx(0.3)
    assert thresholds[1] == pytest.approx(0.5)


def test_attribute_distribution_compares_three_splits(tmp_path):
    _write_attr_csv(
        tmp_path / "train.csv",
        ["a.jpg,1,1", "b.jpg,1,0", "c.jpg,0,0", "d.jpg,0,1"],
    )
    _write_attr_csv(tmp_path / "val.csv", ["e.jpg,1,0", "f.jpg,0,1"])
    _write_attr_csv(tmp_path / "test.csv", ["g.jpg,0,1", "h.jpg,0,1"])
    frame = attribute_distribution_frame(
        {
            "train": [tmp_path / "train.csv"],
            "val": [tmp_path / "val.csv"],
            "test": [tmp_path / "test.csv"],
        }
    ).set_index("attribute")
    assert list(frame.index) == ["Bald", "Male"]
    assert frame.loc["Bald", "train_positive"] == 2
    assert frame.loc["Bald", "train_count"] == 4
    assert frame.loc["Bald", "train_rate"] == 0.5
    assert frame.loc["Bald", "val_rate"] == 0.5
    assert frame.loc["Bald", "test_rate"] == 0
    assert frame.loc["Bald", "val_minus_train"] == 0
    assert frame.loc["Bald", "test_minus_train"] == -0.5
    assert frame.loc["Male", "test_positive"] == 2

    trainer = SimpleNamespace(
        is_global_zero=True,
        log_dir=str(tmp_path / "run"),
        loggers=[],
        logger=None,
        datamodule=SimpleNamespace(
            train_csv_paths=[tmp_path / "train.csv"],
            val_csv_paths=[tmp_path / "val.csv"],
            test_csv_paths=[tmp_path / "test.csv"],
            attribute_columns=None,
        ),
    )
    module = SimpleNamespace(global_step=0, logger=None)
    callback = SaveAttributeDistributionCallback()
    callback.on_fit_start(trainer, module)
    saved = pd.read_csv(tmp_path / "run" / "attribute_distribution.csv")
    assert list(saved["attribute"]) == ["Bald", "Male"]
    assert list(saved.columns[:6]) == [
        "attribute",
        "train_rate",
        "val_rate",
        "test_rate",
        "val_minus_train",
        "test_minus_train",
    ]
    assert (tmp_path / "run" / "attribute_distribution.png").is_file()
    assert cv2.imread(str(tmp_path / "run" / "attribute_distribution.png")) is not None

    trainer.training = True
    (tmp_path / "run" / "attribute_distribution.csv").unlink()
    callback.on_validation_start(trainer, module)
    assert not (tmp_path / "run" / "attribute_distribution.csv").exists()

    trainer.training = False
    callback.on_validation_start(trainer, module)
    callback.on_test_start(trainer, module)
    assert (tmp_path / "run" / "attribute_distribution.csv").is_file()
