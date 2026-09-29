"""分类 test 可视化：错分目录、边框色、混淆矩阵落到 logger 目录。"""

from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lovely_deep_learning.callback.image_classifier import (
    SaveImageClassifierTestPredictVisualizationCallback,
    error_pair_bgr,
    error_pair_dirname,
)


class _Dataset:
    map_class_id_to_class_name = {0: "head-age_0", 1: "head-age_1", 2: "head-age_2"}

    def convert_img_from_tensor_to_numpy(self, img):
        return np.zeros((32, 48, 3), dtype=np.uint8)

    def draw_target_and_predict_label_on_numpy(self, img, **kwargs):
        return img


class _Experiment:
    def __init__(self):
        self.images = []

    def add_image(self, tag, img, global_step=0):
        self.images.append((tag, tuple(img.shape), global_step))


def _sample(path: str, class_id: int, class_name: str, pred_id: int, conf: float = 0.8):
    net_in = [{"img_path": path, "img_tv_transformed": torch.zeros(3, 8, 8)}]
    net_out = {"class_id": torch.tensor([class_id]), "class_name": [class_name]}
    outputs = {
        "metric_preds": {
            "pred_ids": torch.tensor([pred_id]),
            "pred_conf": torch.tensor([conf]),
        }
    }
    return outputs, (net_in, net_out)


def test_error_pair_colors_are_stable_and_distinct():
    pairs = [(g, p) for g in range(3) for p in range(3) if g != p]
    colors = [error_pair_bgr(g, p) for g, p in pairs]
    assert len(set(colors)) == len(pairs)
    assert error_pair_bgr(0, 1) == error_pair_bgr(0, 1)
    assert error_pair_dirname("head-age_0", "head-age_1") == "head-age_0__head-age_1"


def test_test_end_writes_matrix_and_groups_mistakes(tmp_path):
    experiment = _Experiment()
    logger = SimpleNamespace(experiment=experiment, log_dir=str(tmp_path))
    trainer = SimpleNamespace(
        log_dir=str(tmp_path),
        logger=logger,
        loggers=[logger],
        test_dataloaders=SimpleNamespace(dataset=_Dataset()),
    )
    module = SimpleNamespace(global_step=3)
    cb = SaveImageClassifierTestPredictVisualizationCallback(
        save_dir=None, test_only_save_mistake=True
    )

    cb.on_test_batch_end(trainer, module, *_sample("/data/ok.jpg", 0, "head-age_0", 0), 0)
    cb.on_test_batch_end(
        trainer, module, *_sample("/data/crop.jpg", 0, "head-age_0", 1, 0.91), 1
    )
    cb.on_test_batch_end(
        trainer, module, *_sample("/data/crop.jpg", 1, "head-age_1", 2, 0.4), 2
    )
    cb.on_test_end(trainer, module)

    matrix = tmp_path / "confusion_matrix.png"
    assert matrix.is_file()
    assert cv2.imread(str(matrix)) is not None
    mistake_0 = tmp_path / "test" / "head-age_0__head-age_1" / "crop.jpg"
    mistake_1 = tmp_path / "test" / "head-age_1__head-age_2" / "crop.jpg"
    assert mistake_0.is_file()
    assert mistake_1.is_file()
    assert not (tmp_path / "test" / "head-age_0__head-age_0").exists()
    assert experiment.images and experiment.images[0][0] == "test/confusion_matrix"
    assert experiment.images[0][2] == 3

    saved = cv2.imread(str(mistake_0))
    border = error_pair_bgr(0, 1)
    # JPEG 有损，只要求边框接近该错分对的颜色，且和另一错分对不同。
    corner = saved[0, 0].astype(np.int16)
    assert np.abs(corner - np.array(border)).max() < 40
    other = cv2.imread(str(mistake_1))[0, 0].astype(np.int16)
    assert np.abs(corner - other).max() > 40
