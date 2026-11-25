import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import v2

from lovely_deep_learning.data_module.attribute_classifier import AttributeFromDetectBoxDataModule
from lovely_deep_learning.dataset.attribute_classifier import AttributeFromDetectBoxDataset


def _write_image(path, height, width, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return image


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _rows(image_name):
    return [
        {
            "path_img": image_name,
            "cx": 0.5,
            "cy": 0.5,
            "w": 1,
            "h": 1,
            "img_w": 50,
            "img_h": 40,
            "attr_Male": 1,
            "attr_Young": 0,
            "note": "keep",
        },
        {
            "path_img": image_name,
            "cx": 0.2,
            "cy": 0.2,
            "w": 0.05,
            "h": 0.05,
            "img_w": 50,
            "img_h": 40,
            "attr_Male": 0,
            "attr_Young": 1,
            "note": "short",
        },
    ]


def test_selects_columns_in_given_order_and_drops_short_boxes(tmp_path):
    image = tmp_path / "face.jpg"
    _write_image(image, 40, 50, 30)
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, _rows("face.jpg"))

    dataset = AttributeFromDetectBoxDataset(
        [csv_path],
        attribute_columns=["attr_Young", "attr_Male"],
        min_side_px=16,
    )

    assert dataset.attribute_columns == ["attr_Young", "attr_Male"]
    assert len(dataset) == 1
    net_in, net_out = dataset[0]
    assert net_in["img_tv_transformed"].shape == (3, 40, 50)
    assert torch.equal(net_out["attr"], torch.tensor([0.0, 1.0]))
    collated_in, collated_out = dataset.get_collate_fn_for_dataloader()([dataset[0], dataset[0]])
    assert collated_out["attr"].shape == (2, 2)
    assert collated_in[0]["img"].shape == (3, 40, 50)


def test_none_collects_attr_columns_in_header_order(tmp_path):
    image = tmp_path / "face.jpg"
    _write_image(image, 40, 50, 30)
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, _rows("face.jpg"))

    dataset = AttributeFromDetectBoxDataset([csv_path], min_side_px=16)

    assert dataset.attribute_columns == ["attr_Male", "attr_Young"]


def test_full_frame_box_stays_the_whole_image(tmp_path):
    image_path = tmp_path / "face.jpg"
    source = _write_image(image_path, 40, 50, 90)
    csv_path = tmp_path / "train.csv"
    _write_csv(csv_path, [_rows("face.jpg")[0]])

    dataset = AttributeFromDetectBoxDataset([csv_path], box_scale=1.2, min_side_px=0)
    net_in, _net_out = dataset[0]

    assert tuple(net_in["img_shape"]) == source.shape
    assert net_in["img_tv_transformed"].shape[-2:] == (40, 50)


def test_small_box_expands_inside_the_image(tmp_path):
    image_path = tmp_path / "scene.jpg"
    _write_image(image_path, 100, 120, 10)
    csv_path = tmp_path / "train.csv"
    _write_csv(
        csv_path,
        [
            {
                "path_img": "scene.jpg",
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "img_w": 120,
                "img_h": 100,
                "attr_Male": 1,
            }
        ],
    )

    dataset = AttributeFromDetectBoxDataset([csv_path], box_scale=1.2, min_side_px=0)
    _net_in, net_out = dataset[0]
    crop_h, crop_w = dataset[0][0]["img_tv_transformed"].shape[-2:]

    assert crop_h < 100 and crop_w < 120
    assert crop_h > 20 and crop_w > 24
    assert int(net_out["attr"]) == 1


def test_bad_attr_value_and_missing_column_raise(tmp_path):
    image = tmp_path / "face.jpg"
    _write_image(image, 40, 50, 30)
    csv_path = tmp_path / "train.csv"
    rows = _rows("face.jpg")
    rows[0]["attr_Male"] = 2
    _write_csv(csv_path, rows)

    with pytest.raises(ValueError, match="attr_Male"):
        AttributeFromDetectBoxDataset([csv_path], min_side_px=16)

    _write_csv(csv_path, _rows("face.jpg"))
    with pytest.raises(ValueError, match="attr_Smiling"):
        AttributeFromDetectBoxDataset(
            [csv_path],
            attribute_columns=["attr_Smiling"],
            min_side_px=16,
        )


def test_predict_without_attr_columns_has_empty_net_out(tmp_path):
    image = tmp_path / "face.jpg"
    _write_image(image, 40, 50, 30)
    csv_path = tmp_path / "predict.csv"
    _write_csv(
        csv_path,
        [
            {
                "path_img": "face.jpg",
                "cx": 0.5,
                "cy": 0.5,
                "w": 1,
                "h": 1,
                "img_w": 50,
                "img_h": 40,
            }
        ],
    )

    dataset = AttributeFromDetectBoxDataset([csv_path], attribute_columns=[], min_side_px=0)
    _net_in, net_out = dataset[0]
    _collated_in, collated_out = dataset.get_collate_fn_for_dataloader()([dataset[0]])

    assert dataset.attribute_columns == []
    assert net_out == {}
    assert collated_out == {}


def test_datamodule_shares_attribute_columns(tmp_path):
    image = tmp_path / "face.jpg"
    _write_image(image, 40, 50, 30)
    labeled = tmp_path / "train.csv"
    predict = tmp_path / "predict.csv"
    _write_csv(labeled, _rows("face.jpg"))
    _write_csv(
        predict,
        [
            {
                "path_img": "face.jpg",
                "cx": 0.5,
                "cy": 0.5,
                "w": 1,
                "h": 1,
                "img_w": 50,
                "img_h": 40,
            }
        ],
    )
    transform = v2.Compose([v2.Resize(size=(32, 32))])
    module = AttributeFromDetectBoxDataModule(
        train_csv_paths=[labeled],
        val_csv_paths=[labeled],
        test_csv_paths=[labeled],
        predict_csv_paths=[predict],
        transform_train=transform,
        transform_val=transform,
        attribute_columns=["attr_Male"],
        predict_attribute_columns=[],
        min_side_px=16,
        batch_size=1,
        num_workers=0,
    )

    module.setup("fit")
    module.setup("predict")

    assert module.train_dataset.attribute_columns == ["attr_Male"]
    assert module.val_dataset.attribute_columns == ["attr_Male"]
    assert module.pred_dataset.attribute_columns == []
    assert module.train_dataset[0][1]["attr"].shape == (1,)
    assert module.train_dataset[0][0]["img_tv_transformed"].shape[-2:] == (32, 32)
