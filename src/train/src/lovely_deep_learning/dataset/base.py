import csv
import os
import hashlib
from typing import List, Dict, Optional, Callable, Any, Union, Sequence
from torch.utils.data import Dataset
from tqdm import tqdm  # 直接引入tqdm用于进度显示
import cv2
import numpy as np
from pathlib import Path
import torch


class BaseDataset(Dataset):
    """
    基础数据集类，继承自 PyTorch 的 Dataset，用于从 CSV 加载多列路径/标量字段。

    - 相对路径相对于该 CSV 文件所在目录解析；空单元格记为 ""（列名含 path 时计入负样本统计）。
    - 列名包含子串 path 的列按路径解析为绝对路径，不做磁盘存在性检查。
    - CSV 缺少 key_map 中的表头时，该列对该文件各行填 ""。
    - csv_paths 入参可为 str 或 Path；self.csv_paths 恒为已 expanduser+resolve 的 List[Path]。
    """

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
    ):
        self.csv_paths: List[Path] = [Path(p).expanduser().resolve() for p in csv_paths]
        self.key_map = key_map
        self.transform = transform

        self.sample_path_table: Dict[str, List[str]] = {}
        self.num_samples: int = 0
        self._stats_negative_samples: int = 0

        self.sample_path_table = self._generate_sample_path_table()
        self.num_samples = self._count_samples()

    def _generate_sample_path_table(self) -> Dict[str, List[str]]:
        path_table = {inner_field: [] for inner_field in self.key_map.keys()}

        for csv_path in self.csv_paths:
            csv_dir = csv_path.parent

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                for row in reader:
                    has_empty_path = False
                    current_row: Dict[str, str] = {}

                    for inner_field, csv_field in self.key_map.items():
                        if csv_field not in fieldnames:
                            raw_value = ""
                        else:
                            raw_value = (row.get(csv_field) or "").strip()

                        is_path_field = "path" in csv_field.lower()

                        if not raw_value:
                            current_row[inner_field] = ""
                            if is_path_field:
                                has_empty_path = True
                            continue

                        if is_path_field:
                            if os.path.isabs(raw_value):
                                resolved_path = raw_value
                            else:
                                resolved_path = str((csv_dir / raw_value).resolve())
                            current_row[inner_field] = resolved_path
                        else:
                            current_row[inner_field] = raw_value

                    for field, value in current_row.items():
                        path_table[field].append(value)

                    if has_empty_path:
                        self._stats_negative_samples += 1

        return path_table

    def _count_samples(self) -> int:
        if not self.sample_path_table:
            return 0
        return len(next(iter(self.sample_path_table.values())))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, str]:
        return {
            inner_field: self.sample_path_table[inner_field][index] for inner_field in self.key_map.keys()
        }

    def __str__(self) -> str:
        negative_ratio = (self._stats_negative_samples / self.num_samples * 100) if self.num_samples > 0 else 0

        lines = [
            "=" * 70,
            "📊 BaseDataset 统计",
            "-" * 70,
            f"1. CSV 文件（{len(self.csv_paths)} 个）：",
        ]
        for i, path in enumerate(self.csv_paths, 1):
            lines.append(f"   {i}. {path}")

        lines.append(f"\n2. 字段映射（{len(self.key_map)} 个）：")
        for inner_field, csv_field in self.key_map.items():
            lines.append(f"   [{inner_field}] → [{csv_field}]")

        lines.extend(
            [
                f"\n3. 样本数：{self.num_samples}",
                f"   负样本行（含空 path 列）：{self._stats_negative_samples}（{negative_ratio:.1f}%）",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    def draw_target_and_predict_label_on_numpy(self):
        pass

    @staticmethod
    def cache_image(img_paths: List[str], cache_dir: str) -> List[str]:
        """
        生成图像缓存（仅加速读取，不做任何预处理）

        功能：
            将原始图像以.npy格式保存到指定目录，通过图像内容哈希确保缓存唯一性，
            已存在的缓存会被复用，最终返回与输入图像路径顺序一致的缓存路径列表。

        缓存文件名规则：
            采用图像内容MD5哈希前16位 + ".npy"格式，例如：
            "a1b2c3d4e5f6g7h8.npy"，便于区分图像缓存与其他类型文件。

        参数：
            img_paths: List[str] - 原始图像路径列表（完整路径）
            cache_dir: str - 缓存文件保存目录（不存在时自动创建）

        返回：
            List[str] - 与img_paths顺序对应的.npy缓存路径列表

        异常：
            FileNotFoundError: 当输入图像路径不存在或无法读取时抛出
        """
        # 确保缓存目录存在，不存在则创建
        os.makedirs(cache_dir, exist_ok=True)

        # 存储缓存路径，与输入图像路径顺序严格一致
        npy_paths = []

        # 遍历所有图像路径，带进度条显示
        for img_path in tqdm(img_paths, desc="生成图像缓存", unit="张"):
            # 生成图像内容的MD5哈希（确保相同图像复用缓存）
            with open(img_path, "rb") as f:
                img_content = f.read()
            img_hash = hashlib.md5(img_content).hexdigest()[:16]  # 取前16位哈希值

            # 构建缓存文件名和完整路径
            npy_filename = f"{img_hash}.npy"
            npy_path = os.path.join(cache_dir, npy_filename)

            # 若缓存不存在，则生成（仅读取原图，不做任何预处理）
            if not os.path.exists(npy_path):
                # 读取原始图像（保留OpenCV默认的BGR通道顺序）
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"无法读取图像文件（路径不存在或文件损坏）：{img_path}")

                # 直接保存原始图像数据（不做通道转换、resize等任何操作）
                np.save(npy_path, img)

            # 记录当前图像的缓存路径，保持与输入顺序一致
            npy_paths.append(npy_path)

        return npy_paths

    @staticmethod
    def get_hash(obj: Any) -> str:
        """生成任意对象的MD5哈希值，用于缓存校验"""
        hash_obj = hashlib.md5()

        if isinstance(obj, (list, tuple)):
            # 对列表/元组，排序后序列化（确保顺序不影响哈希）
            for item in sorted(obj):
                hash_obj.update(str(item).encode("utf-8"))
        elif isinstance(obj, dict):
            # 对字典，按键排序后序列化
            for key in sorted(obj):
                hash_obj.update(f"{key}:{obj[key]}".encode("utf-8"))
        elif isinstance(obj, (str, Path)):
            # 对文件路径，读取文件内容生成哈希
            obj = Path(obj)
            if obj.exists() and obj.is_file():
                # 读取前1MB内容（平衡效率和准确性）
                with open(obj, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        hash_obj.update(chunk)
                        break  # 仅读1MB
        else:
            # 其他类型直接序列化
            hash_obj.update(str(obj).encode("utf-8"))

        return hash_obj.hexdigest()

    @staticmethod
    def read_img(img_path: str, img_npy_path: Union[str, None]):
        if img_npy_path:
            img = np.load(img_npy_path)
        else:
            img = cv2.imread(img_path)
        return img, img.shape

    def convert_img_from_tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
        pass

    def convert_img_from_numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
        pass

    @staticmethod
    def convert_img_from_numpy_to_tensor_uint8(img: np.ndarray) -> torch.Tensor:
        """
        将numpy格式的图像转换为RGB格式的tensor，并调整维度从(H, W, C)到(C, H, W)

        Args:
            img: 输入的numpy图像数组

        Returns:
            转换后的torch.Tensor，格式为(C, H, W)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 转为 torch.Tensor 并调整维度 (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)  # (C, H, W)
        img_tensor = img_tensor.contiguous()

        return img_tensor

    @staticmethod
    def convert_img_from_tensor_to_numpy_uint8(img: torch.Tensor) -> np.ndarray:
        """
        将RGB格式的tensor转换为numpy格式的图像，并调整维度从(C, H, W)到(H, W, C)

        Args:
            img: 输入的torch.Tensor，格式为(C, H, W)

        Returns:
            转换后的numpy图像数组，格式为(H, W, C)
        """
        img = img.detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()  # CHW -> HWC
        img_np = img_np.astype(np.uint8)

        # RGB -> BGR (opencv)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_np

    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(x):
            return list(zip(*x))


if __name__ == "__main__":
    # 示例用法

    # 假设CSV文件和数据文件夹结构如下：
    # dataset/
    #   ├─ data.csv
    #   ├─ images/
    #   │   ├─ 001.jpg
    #   │   └─ 002.jpg
    #   └─ labels/
    #       ├─ 001.txt
    #       └─ 002.txt

    CSV_FILES = [
        Path("/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/train.csv")
    ]
    FIELD_MAP = {
        "img_path": "data_img",  # 类内字段img对应CSV中的image_path列
        "label_path": "label_detect_yolo",  # 类内字段label对应CSV中的label_path列
    }

    dataset = BaseDataset(csv_paths=CSV_FILES, key_map=FIELD_MAP)

    dataset.cache_image(dataset.sample_path_table["img_path"], "cache")
    print(dataset)
