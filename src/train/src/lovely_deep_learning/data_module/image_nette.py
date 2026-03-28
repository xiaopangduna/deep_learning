from pathlib import Path
from typing import List

import urllib.request
import tarfile
import pandas as pd
from .image_classifier import ImageClassifierDataModule


class ImageNetteDataModule(ImageClassifierDataModule):
    """ImageNette 2-320：下载数据、生成 CSV，并沿用图像分类 DataModule 的训练 / 验证 / 测试 / 预测流程。

    使用顺序
    --------
    1. 构造时传入 ``dataset_dir`` 以及各阶段 CSV 路径、``transform_*``、``key_map`` 等（见下方参数）。
    2. ``trainer.fit`` 之前 Lightning 会调用 ``prepare_data``：若 ``dataset_dir`` 不存在则下载并解压，
       然后在 ``dataset_dir.parent`` 下写入 train/val/predict 三个 CSV 及 ``map_class_id_to_class_name.csv``
       （若主 CSV 已存在则跳过整批生成；仅缺映射文件时会补写映射）。
    3. ``setup`` 由父类完成，按 CSV 构建 ``ImageClassifierDataset``。

    路径约定
    --------
    - ``dataset_dir`` 默认为 ``datasets/IMAGENETTE/imagenette2-320``，即内含 ``train/``、``val/``
      类别子目录的根目录（与 fast.ai 发布的目录结构一致）。
    - 生成的 CSV 中 ``path_img`` 为相对路径（如 ``imagenette2-320/train/...``），相对 **CSV 文件所在目录**
      解析；因此 YAML 里 ``train_csv_paths`` 等应列为 ``datasets/IMAGENETTE/train.csv`` 这类与解压布局
      一致的路径。

    构造参数见 ``__init__``（显式列出 ``BaseDataModule`` / ``ImageClassifierDataModule`` 的全部字段，
    便于类型检查与 IDE 补全）。若提供 ``map_class_id_to_class_name``，生成 CSV 时的类别顺序与 id
    以该映射为准；否则按 ``train`` 下类别文件夹名字典序分配 id。
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
        dataset_dir="datasets/IMAGENETTE/imagenette2-320",
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
            图像分类 DataModule 与 Dataset 共用配置；语义见 ``ImageClassifierDataModule``。
        dataset_dir : str or pathlib.Path
            ImageNette 解压后的根目录（含 ``train/``、``val/``）。tar 下载至其父目录，CSV 亦写在父目录下。
        """
        super().__init__(
            key_map=key_map,
            predict_key_map=predict_key_map,
            map_class_id_to_class_name=map_class_id_to_class_name,
            norm_mean=norm_mean,
            norm_std=norm_std,
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

    def _class_name_to_idx(self, dataset_dir: Path) -> dict[str, int]:
        m = self._map_spec_effective_for_csv_generation(
            self._map_class_id_to_class_name_spec
        )
        if m:
            return {name: idx for idx, name in m.items()}
        classes = sorted(
            d.name for d in (dataset_dir / "train").iterdir() if d.is_dir()
        )
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def prepare_data(self):
        """下载并解压 ImageNette 2-320（若 ``dataset_dir`` 尚不存在），再生成或跳过已存在的 CSV 与映射文件。

        解压目录为 ``dataset_dir`` 的父目录；``self.dataset_dir`` 指向含 ``train``/``val`` 的数据根。
        """
        parent_dir = self.dataset_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        imagenette_tar = parent_dir / "imagenette2-320.tgz"

        if not self.dataset_dir.exists():
            print(f"Downloading ImageNette dataset to {imagenette_tar} ...")
            urllib.request.urlretrieve(imagenette_url, str(imagenette_tar))

            print(f"Extracting {imagenette_tar} ...")
            with tarfile.open(imagenette_tar, "r:gz") as tar:
                tar.extractall(path=str(parent_dir))  # 解压到父目录

            print("Download and extraction complete.")
        else:
            print("ImageNette dataset already exists, skipping download.")

        # 生成CSV文件
        self._generate_csv_files(self.dataset_dir)

    def _generate_csv_files(self, dataset_dir: Path):
        """
        生成 train.csv、val.csv、predict.csv 及 map_class_id_to_class_name.csv。
        Args:
            dataset_dir: 包含图像数据的目录路径
        """
        dataset_dir = Path(dataset_dir)
        output_dir = dataset_dir.parent  # 修改输出目录路径
        output_dir.mkdir(exist_ok=True)

        train_csv_path = output_dir / "train.csv"
        val_csv_path = output_dir / "val.csv"
        predict_csv_path = output_dir / "predict.csv"
        map_csv_path = output_dir / "map_class_id_to_class_name.csv"
        train_split_dir = dataset_dir / "train"

        if (
            train_csv_path.exists()
            and val_csv_path.exists()
            and predict_csv_path.exists()
        ):
            if train_split_dir.is_dir() and not map_csv_path.exists():
                class_to_idx = self._class_name_to_idx(dataset_dir)
                self._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)
                print(f"ImageNetteDataModule: wrote missing {map_csv_path}")
            print("============================================================================")
            print("ImageNetteDataModule: CSV files already exist, skipping generation")
            print(f"  - {train_csv_path}")
            print(f"  - {val_csv_path}")
            print(f"  - {predict_csv_path}")
            print(f"  - {map_csv_path}")
            print("============================================================================")
            return

        class_to_idx = self._class_name_to_idx(dataset_dir)

        # 生成训练集CSV
        train_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "train" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        train_rows.append({
                            'path_img': f"imagenette2-320/train/{class_name}/{img_path.name}",
                            'class_name': class_name,
                            'class_id': class_to_idx[class_name]
                        })
        
        train_df = pd.DataFrame(train_rows, columns=['path_img', 'class_name', 'class_id'])
        train_df.to_csv(train_csv_path, index=False)

        # 生成验证集CSV
        val_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "val" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        val_rows.append({
                            'path_img': f"imagenette2-320/val/{class_name}/{img_path.name}",
                            'class_name': class_name,
                            'class_id': class_to_idx[class_name]
                        })

        val_df = pd.DataFrame(val_rows, columns=['path_img', 'class_name', 'class_id'])
        val_df.to_csv(val_csv_path, index=False)

        # 生成预测集CSV（通常是从验证集中选择的图像，不含标签信息）
        predict_rows = []
        for class_name in class_to_idx.keys():
            class_path = dataset_dir / "val" / class_name
            if class_path.exists():
                for img_path in class_path.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        predict_rows.append({
                            'path_img': f"imagenette2-320/val/{class_name}/{img_path.name}"
                        })

        predict_df = pd.DataFrame(predict_rows, columns=['path_img'])
        predict_df.to_csv(predict_csv_path, index=False)
        self._write_map_class_id_to_class_name_csv(map_csv_path, class_to_idx)

        print("============================================================================")
        print(f"Generated {str(train_csv_path)} with {len(train_df)} entries")
        print(f"Generated {str(val_csv_path)} with {len(val_df)} entries")
        print(f"Generated {str(predict_csv_path)} with {len(predict_rows)} entries")
        print(f"Generated {str(map_csv_path)} with {len(class_to_idx)} classes")
        print("============================================================================")
