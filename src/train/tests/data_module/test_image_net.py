import textwrap
from pathlib import Path

import pytest
import torch
from torchvision.transforms import v2

from lovely_deep_learning.data_module.image_net import ImageNetDataModule

_TRAIN_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_DATASET = _TRAIN_ROOT / "datasets" / "IMAGENET"

_HAS_IMAGENET_LAYOUT = (
    (IMAGENET_DATASET / "ILSVRC/Data/CLS-LOC/train").is_dir()
    and (IMAGENET_DATASET / "ILSVRC/Data/CLS-LOC/val").is_dir()
    and (IMAGENET_DATASET / "ILSVRC/Data/CLS-LOC/test").is_dir()
    and (IMAGENET_DATASET / "ILSVRC/Annotations/CLS-LOC/val").is_dir()
    and (IMAGENET_DATASET / "ILSVRC/ImageSets/CLS-LOC/train_cls.txt").is_file()
    and (IMAGENET_DATASET / "ILSVRC/ImageSets/CLS-LOC/val.txt").is_file()
    and (IMAGENET_DATASET / "ILSVRC/ImageSets/CLS-LOC/test.txt").is_file()
)

KEY_MAP = {"img_path": "path_img", "class_name": "class_name", "class_id": "class_id"}
TRANSFORM = v2.Compose(
    [v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)]
)


def test_first_synset_from_val_xml(tmp_path: Path) -> None:
    xml_path = tmp_path / "ILSVRC2012_val_00000001.xml"
    xml_path.write_text(
        textwrap.dedent(
            """\
            <annotation>
                <object>
                    <name>n01751748</name>
                </object>
            </annotation>
            """
        ),
        encoding="utf-8",
    )
    assert ImageNetDataModule._first_synset_from_val_xml(xml_path) == "n01751748"


def test_first_synset_from_val_xml_multiple_objects(tmp_path: Path) -> None:
    xml_path = tmp_path / "x.xml"
    xml_path.write_text(
        textwrap.dedent(
            """\
            <annotation>
                <object><name>n11111111</name></object>
                <object><name>n22222222</name></object>
            </annotation>
            """
        ),
        encoding="utf-8",
    )
    assert ImageNetDataModule._first_synset_from_val_xml(xml_path) == "n11111111"


@pytest.mark.skipif(not _HAS_IMAGENET_LAYOUT, reason="Local ILSVRC ImageNet layout not found under datasets/IMAGENET")
class TestImageNetDataModule:
    def create_datamodule(self) -> ImageNetDataModule:
        root = str(IMAGENET_DATASET)
        csv_train = str(IMAGENET_DATASET / "train.csv")
        csv_val = str(IMAGENET_DATASET / "val.csv")
        csv_test = str(IMAGENET_DATASET / "test.csv")
        csv_predict = str(IMAGENET_DATASET / "predict.csv")
        return ImageNetDataModule(
            dataset_dir=root,
            train_csv_paths=[csv_train],
            val_csv_paths=[csv_val],
            test_csv_paths=[csv_test],
            predict_csv_paths=[csv_predict],
            key_map=KEY_MAP,
            map_class_id_to_class_name=None,
            batch_size=8,
            num_workers=0,
            transform_train=TRANSFORM,
            transform_val=TRANSFORM,
            transform_test=TRANSFORM,
            transform_predict=TRANSFORM,
        )

    def test_prepare_data(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()

    def test_setup(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()
        dm.setup(stage="fit")

        assert dm.train_dataset is not None, "train_dataset 应该被初始化"
        assert dm.val_dataset is not None, "val_dataset 应该被初始化"

    def test_train_dataloader(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()
        dm.setup(stage="fit")

        loader = dm.train_dataloader()
        net_in, net_out = next(iter(loader))

        assert "img_tv_transformed" in net_in
        assert "class_id" in net_out
        assert net_in["img_tv_transformed"].shape == (dm.batch_size, 3, 224, 224)
        assert net_out["class_id"].shape == (dm.batch_size,)

    def test_val_dataloader(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()
        dm.setup(stage="fit")

        loader = dm.val_dataloader()
        net_in, net_out = next(iter(loader))

        assert "img_tv_transformed" in net_in
        assert "class_id" in net_out
        assert net_in["img_tv_transformed"].shape == (dm.batch_size, 3, 224, 224)
        assert net_out["class_id"].shape == (dm.batch_size,)

    def test_test_dataloader(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()
        dm.setup(stage="test")

        loader = dm.test_dataloader()
        net_in, net_out = next(iter(loader))

        assert "img_tv_transformed" in net_in
        assert net_in["img_tv_transformed"].shape == (dm.batch_size, 3, 224, 224)
        assert net_out == {}

    def test_predict_dataloader(self) -> None:
        dm = self.create_datamodule()
        dm.prepare_data()
        dm.setup(stage="predict")

        loader = dm.predict_dataloader()
        net_in, net_out = next(iter(loader))

        assert "img_tv_transformed" in net_in
        assert net_in["img_tv_transformed"].shape == (dm.batch_size, 3, 224, 224)
        assert net_out == {}
