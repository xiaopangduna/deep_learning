from pathlib import Path
from typing import List
import urllib.request
import zipfile

import pandas as pd

from .object_detect import ObjectDetectDataModule


# COCO 80 类名称（与 YOLO/COCO 类别 id 0–79 一致）
COCO80_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO8_DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
)


class COCO8DataModule(ObjectDetectDataModule):
    """Ultralytics COCO8：下载并解压数据、生成 CSV，并沿用目标检测 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. ``trainer.fit`` 之前 Lightning 会调用 ``prepare_data``：若 ``dataset_dir/images/train`` 等尚不存在则
       从 ``download_url``（默认 ``COCO8_DOWNLOAD_URL``）下载 zip 并解压到 ``dataset_dir`` 的父目录，
       然后在父目录下写入 ``train.csv`` / ``val.csv`` / ``predict.csv`` 及 ``map_class_id_to_class_name.csv``
       （若三个主 CSV 已存在则跳过整批生成；仅缺映射文件时会补写映射）。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ObjectDetectDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/COCO8/coco8``，即解压后含 ``images/train``、``images/val``、
      ``labels/train``、``labels/val`` 的根目录（与官方 zip 内 ``coco8/`` 布局一致）。
    - 生成的 CSV 中 ``path_img``、``path_label_detect_yolo`` 为相对路径（如 ``coco8/images/train/...``），
      相对 **CSV 文件所在目录** 解析；因此 YAML 里 ``train_csv_paths`` 等应列为 ``datasets/COCO8/train.csv``
      这类与解压布局一致的路径。
    - ``predict.csv`` 仅含 ``path_img``，默认取验证集图像（与 ``ImageNetteDataModule`` 一致）。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ObjectDetectDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。``map_class_id_to_class_name`` 在 ``setup`` 中的解析语义见父类；
    ``prepare_data`` 写出的类别映射表为 COCO 80 类标准名称与 id。
    """

    def __init__(
        self,
        train_csv_paths: List[str],
        val_csv_paths: List[str],
        test_csv_paths: List[str],
        predict_csv_paths: List[str],
        transform_train,
        transform_val,
        transform_test=None,
        transform_predict=None,
        batch_size: int = 32,
        num_workers: int = 4,
        key_map=None,
        predict_key_map=None,
        map_class_id_to_class_name=None,
        norm_mean=None,
        norm_std=None,
        nms: bool = True,
        nms_iou: float = 0.7,
        inference_conf_thres: float = 0.001,
        dataset_dir: str = "datasets/COCO8/coco8",
        download_url: str | None = None,
    ):
        """
        Parameters
        ----------
        train_csv_paths, val_csv_paths, test_csv_paths, predict_csv_paths
            各阶段 CSV 路径列表；须与 ``prepare_data`` 生成文件位置一致。详见 ``BaseDataModule``。
        transform_train, transform_val
            训练、验证集变换。详见 ``BaseDataModule``。
        transform_test, transform_predict
            测试、预测变换；默认 ``None`` 时父类回退为 ``transform_val``。
        batch_size, num_workers
            各 ``DataLoader`` 批量与 worker 数。
        key_map, predict_key_map, map_class_id_to_class_name, norm_mean, norm_std
            目标检测 DataModule 与 Dataset 共用配置；语义见 ``ObjectDetectDataModule``。
        nms, nms_iou, inference_conf_thres
            推理后处理参数，传给 ``ObjectDetectDataset`` 并在 Module 中复用。
        dataset_dir : str or pathlib.Path
            COCO8 解压后的根目录（含 ``images/``、``labels/``）。zip 下载至其父目录并解压到该父目录下。
        download_url
            数据集 zip 地址；默认 ``COCO8_DOWNLOAD_URL``（Ultralytics 官方 coco8.zip）。
        """
        super().__init__(
            key_map=key_map,
            predict_key_map=predict_key_map,
            map_class_id_to_class_name=map_class_id_to_class_name,
            norm_mean=norm_mean,
            norm_std=norm_std,
            nms=nms,
            nms_iou=nms_iou,
            inference_conf_thres=inference_conf_thres,
            train_csv_paths=train_csv_paths,
            val_csv_paths=val_csv_paths,
            test_csv_paths=test_csv_paths,
            predict_csv_paths=predict_csv_paths,
            transform_train=transform_train,
            transform_val=transform_val,
            transform_test=transform_test,
            transform_predict=transform_predict,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.dataset_dir = Path(dataset_dir)
        self.download_url = download_url or COCO8_DOWNLOAD_URL

    def prepare_data(self):
        """若本地尚无 COCO8 目录则下载并解压 zip，再生成或跳过已存在的 CSV 与映射文件。"""
        parent_dir = self.dataset_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        train_img_dir = self.dataset_dir / "images" / "train"
        val_img_dir = self.dataset_dir / "images" / "val"

        if not (train_img_dir.is_dir() and val_img_dir.is_dir()):
            zip_path = parent_dir / "coco8.zip"
            print(f"Downloading COCO8 dataset to {zip_path} ...")
            urllib.request.urlretrieve(self.download_url, str(zip_path))

            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(path=str(parent_dir))
            zip_path.unlink(missing_ok=True)
            print("Download and extraction complete.")
        else:
            print("COCO8 dataset already exists, skipping download.")

        self._generate_csv_files(self.dataset_dir)

    @staticmethod
    def _coco_class_name_to_idx() -> dict[str, int]:
        return {name: i for i, name in enumerate(COCO80_CLASS_NAMES)}

    def _collect_split_rows(self, dataset_dir: Path, split: str) -> list[dict]:
        """``split`` 为 ``train`` 或 ``val``：生成带 ``path_img``、``path_label_detect_yolo`` 的行。"""
        prefix = dataset_dir.name
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        rows: list[dict] = []
        if not img_dir.is_dir():
            return rows
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.is_file():
                continue
            rows.append(
                {
                    "path_img": f"{prefix}/images/{split}/{img_path.name}",
                    "path_label_detect_yolo": f"{prefix}/labels/{split}/{stem}.txt",
                }
            )
        return rows

    def _generate_csv_files(self, dataset_dir: Path):
        """生成 ``train.csv``、``val.csv``、``predict.csv`` 及 ``map_class_id_to_class_name.csv``。"""
        dataset_dir = Path(dataset_dir)
        output_dir = dataset_dir.parent
        output_dir.mkdir(exist_ok=True)

        train_csv_path = output_dir / "train.csv"
        val_csv_path = output_dir / "val.csv"
        predict_csv_path = output_dir / "predict.csv"
        map_csv_path = output_dir / "map_class_id_to_class_name.csv"
        train_split_dir = dataset_dir / "images" / "train"

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and predict_csv_path.exists()
        ):
            if train_split_dir.is_dir() and not map_csv_path.exists():
                class_to_idx = self._coco_class_name_to_idx()
                self._write_map_class_id_to_class_name_csv(
                    map_csv_path, class_to_idx)
                print(f"COCO8DataModule: wrote missing {map_csv_path}")
            print(
                "============================================================================")
            print("COCO8DataModule: CSV files already exist, skipping generation")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {predict_csv_path}")
            print(f"  - {map_csv_path}")
            print(
                "============================================================================")
            return

        class_to_idx = self._coco_class_name_to_idx()
        train_rows = self._collect_split_rows(dataset_dir, "train")
        val_rows = self._collect_split_rows(dataset_dir, "val")

        predict_rows = [{"path_img": r["path_img"]} for r in val_rows]

        pd.DataFrame(
            train_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(train_csv_path, index=False)
        pd.DataFrame(
            val_rows, columns=["path_img", "path_label_detect_yolo"]
        ).to_csv(val_csv_path, index=False)
        pd.DataFrame(predict_rows, columns=["path_img"]).to_csv(
            predict_csv_path, index=False
        )
        self._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

        print(
            "============================================================================")
        print(f"Generated {train_csv_path} with {len(train_rows)} entries")
        print(f"Generated {val_csv_path} with {len(val_rows)} entries")
        print(f"Generated {predict_csv_path} with {len(predict_rows)} entries")
        print(f"Generated {map_csv_path} with {len(class_to_idx)} classes")
        print(
            "============================================================================")
