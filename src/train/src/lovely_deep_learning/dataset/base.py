import csv
import os
import hashlib
from typing import List, Dict, Optional, Callable, Any,Union
from torch.utils.data import Dataset
from tqdm import tqdm  # 直接引入tqdm用于进度显示
import cv2
import numpy as np
import hashlib
from pathlib import Path
import torch

class BaseDataset(Dataset):
    """
    基础数据集类，继承自PyTorch的Dataset，用于加载和管理带路径信息的CSV格式数据集。

    【核心功能】
    - 解析CSV中的绝对/相对路径（相对路径基于CSV所在目录解析）
    - 区分有效样本（含负样本）与无效样本，仅保留有效样本
    - 提供完整的样本统计信息（总数、负样本占比、无效样本详情等）

    【核心属性】
    - csv_paths: List[str]，输入的CSV文件路径列表
    - key_map: Dict[str, str]，类内字段→CSV表头字段的映射（如{"img": "image_path"}）
    - transform: Optional[Callable]，数据增强实例（外部传入，需为可调用对象，如Compose包装的增强流水线；默认None，即不增强）
    - sample_path_table: Dict[str, List[str]]，路径表格（键为类内字段，值为路径列表，空字符串表示负样本）
    - num_samples: int，有效样本总数（含负样本）

    【核心方法】
    - __init__: 初始化数据集，完成输入验证、路径解析和样本计数
    - __getitem__: 通过索引获取样本容器（字典，键为"类内字段_path"）
    - __len__: 返回有效样本总数
    - __str__: 打印数据集完整统计信息（CSV列表、字段映射、样本统计等）

    【用法示例】
    >>> # 1. 定义字段映射（类内字段→CSV表头）
    >>> key_map = {"img": "image_path", "label": "label_path"}
    >>> # 2. 实例化数据集
    >>> dataset = BaseDataset(csv_paths=["train.csv"], key_map=key_map)
    >>> # 3. 查看统计信息
    >>> print(dataset)
    >>> # 4. 访问样本（返回{"img_path": "...", "label_path": "..."}）
    >>> sample = dataset[0]

    【术语说明】
    - 负样本：包含空路径（""）的有效样本（如label_path为空，表示无标签的负样本）
    - 无效样本：包含非空但实际不存在的路径的样本（会被过滤，不纳入有效样本）
    """

    def __init__(self, csv_paths: List[str], key_map: Dict[str, str], transform: Optional[Callable] = None):
        # 核心配置参数
        self.csv_paths = csv_paths  # CSV文件路径列表
        self.key_map = key_map  # 类内字段→CSV表头字段的映射
        self.transform = transform  # 

        # 核心数据
        self.sample_path_table: Dict[str, List[str]] = {}  # 存储解析后的绝对路径
        self.num_samples: int = 0

        # 统计信息
        self._stats_invalid_records: List[str] = []  # 记录无效路径详情
        self._stats_negative_samples: int = 0  # 新增：负样本计数（含空路径的样本）
        # 初始化
        self._validate_inputs()
        self.sample_path_table = self._generate_sample_path_table()
        self.num_samples = self._count_and_validate_samples()

    def _validate_inputs(self) -> None:
        """验证输入的CSV文件是否存在"""
        if not isinstance(self.csv_paths, list) or len(self.csv_paths) == 0:
            raise ValueError("❌ csv_paths必须是至少包含1个文件路径的列表")

        for path in self.csv_paths:
            abs_path = os.path.abspath(path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"❌ CSV文件不存在：{path}（解析为绝对路径：{abs_path}）")

        if not isinstance(self.key_map, dict) or len(self.key_map) == 0:
            raise ValueError("❌ key_map必须是至少包含1个键值对的字典")

        # 检查CSV表头字段是否重复
        csv_fields = list(self.key_map.values())
        if len(csv_fields) != len(set(csv_fields)):
            duplicates = [f for f in set(csv_fields) if csv_fields.count(f) > 1]
            raise ValueError(f"❌ key_map中CSV表头字段重复：{duplicates}")

    def _generate_sample_path_table(self) -> Dict[str, List[str]]:
        """生成路径表格，正确处理绝对路径、相对路径和空路径（负样本）"""
        path_table = {inner_field: [] for inner_field in self.key_map.keys()}
        csv_fields = list(self.key_map.values())

        for csv_path in self.csv_paths:
            # 1. 解析CSV文件的绝对路径和所在目录
            abs_csv_path = os.path.abspath(csv_path)
            csv_dir = os.path.dirname(abs_csv_path)  # 相对路径的基准目录

            with open(abs_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                # 检查CSV是否包含所需字段 - 现在改为记录缺失字段而非报错
                missing_fields = [f for f in csv_fields if f not in reader.fieldnames]
                if missing_fields:
                    print(f"⚠️  CSV {abs_csv_path} 缺少字段：{missing_fields}，将忽略这些列")
                    
                    # 找到存在于CSV中的字段
                    present_fields = {k: v for k, v in self.key_map.items() if v in reader.fieldnames}
                else:
                    present_fields = self.key_map

                # 为present_fields创建临时的current_path_table
                current_path_table = {inner_field: [] for inner_field in present_fields.keys()}

                # 2. 逐行处理路径
                for row_idx, row in enumerate(reader, start=2):  # 行号从2开始（表头为1）
                    current_row = {}
                    valid = True
                    error_details = []
                    has_empty_path = False  # 标记当前行是否包含空路径（负样本）

                    for inner_field, csv_field in present_fields.items():
                        # 读取原始值
                        raw_value = row[csv_field].strip()
                        
                        # 检查字段名是否包含"path"，决定是否需要路径验证
                        is_path_field = "path" in csv_field.lower()
                        
                        if not raw_value:
                            # 空值：视为负样本特征，保留空字符串
                            current_row[inner_field] = ""
                            if is_path_field:
                                has_empty_path = True
                            continue  # 空值不影响样本有效性
                        
                        if is_path_field:
                            # 是路径字段，需要解析和验证路径
                            
                            # 3. 路径解析核心逻辑（非空路径）
                            if os.path.isabs(raw_value):
                                resolved_path = raw_value
                            else:
                                resolved_path = os.path.join(csv_dir, raw_value)
                                resolved_path = os.path.abspath(resolved_path)

                            # 4. 检查路径有效性（非空路径必须存在）
                            current_row[inner_field] = resolved_path
                            if not os.path.exists(resolved_path):
                                valid = False
                                error_details.append(
                                    f"字段[{csv_field}]路径不存在（原始路径：{raw_value}，解析后：{resolved_path}）"
                                )
                        else:
                            # 非路径字段，直接存储原始值，不做验证
                            current_row[inner_field] = raw_value

                    # 5. 处理当前行结果
                    if valid:
                        # 所有非空路径都有效：添加到表格
                        for field, path in current_row.items():
                            current_path_table[field].append(path)
                        # 统计负样本（包含空路径的有效样本）
                        if has_empty_path:
                            self._stats_negative_samples += 1
                    else:
                        # 存在无效路径（非空且不存在）：记录并跳过
                        self._stats_invalid_records.append(
                            f"CSV {abs_csv_path} 第{row_idx}行：{'; '.join(error_details)}"
                        )
                
                # 合并当前CSV的结果到主path_table，仅对存在的字段进行合并
                for field in current_path_table:
                    path_table[field].extend(current_path_table[field])
        
        # 删除没有数据的列（即CSV中缺失的字段）
        fields_to_remove = []
        for field in path_table:
            if len(path_table[field]) == 0 and field not in [k for k, v in self.key_map.items() if v in sum([csv.DictReader(open(csv_p)).fieldnames for csv_p in self.csv_paths], [])]:
                # 检查字段是否在任一CSV中存在
                field_exists_in_any_csv = False
                for csv_path in self.csv_paths:
                    with open(csv_path, "r", encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        if self.key_map[field] in reader.fieldnames:
                            field_exists_in_any_csv = True
                            break
                
                if not field_exists_in_any_csv:
                    fields_to_remove.append(field)
        
        for field in fields_to_remove:
            del path_table[field]

        # 打印跳过的样本统计
        total_skipped = len(self._stats_invalid_records)
        if total_skipped > 0:
            print(f"⚠️  共跳过{total_skipped}个无效样本")

        return path_table

    def _count_and_validate_samples(self) -> int:
        """验证所有字段的样本数量是否一致"""
        if not self.sample_path_table:
            return 0

        field_lengths = [len(paths) for paths in self.sample_path_table.values()]
        if len(set(field_lengths)) != 1:
            raise ValueError(f"❌ 字段样本数不匹配：{field_lengths}")

        return field_lengths[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, str]:
        if not 0 <= index < self.num_samples:
            raise IndexError(f"❌ 索引超出范围（有效范围：0~{self.num_samples - 1}）")

        # 生成样本信息容器：键为key_map的类内字段+"_path"，值为对应的绝对路径
        sample_container = {
            f"{inner_field}": self.sample_path_table[inner_field][index] for inner_field in self.key_map.keys()
        }

        return sample_container

    def __str__(self) -> str:
        """
        魔法函数：print(dataset) 时自动调用，输出完整的统计信息
        包含CSV文件信息、字段映射、样本数量及无效样本详情
        """
        # 计算负样本占比
        negative_ratio = (self._stats_negative_samples / self.num_samples * 100) if self.num_samples > 0 else 0

        lines = [
            "=" * 70,
            "📊 BaseDataset 完整统计信息",
            "-" * 70,
        ]

        # 1. CSV文件信息
        lines.append(f"1. 加载的CSV文件（共{len(self.csv_paths)}个）：")
        for i, path in enumerate(self.csv_paths, 1):
            lines.append(f"   {i}. 路径：{path}")

        # 2. 字段映射关系
        lines.append(f"\n2. 字段映射关系（共{len(self.key_map)}个）：")
        for inner_field, csv_field in self.key_map.items():
            lines.append(f"   类内字段[{inner_field}] → CSV表头[{csv_field}]")

        # 3. 样本数量统计（新增负样本信息）
        lines.extend(
            [
                f"\n3. 样本数量统计：",
                f"   有效样本总数（含负样本）：{self.num_samples}",
                f"   负样本数（含空路径）：{self._stats_negative_samples}（{negative_ratio:.1f}%）",
                f"   无效样本总数（非空路径不存在）：{len(self._stats_invalid_records)}",
            ]
        )

        # 4. 无效样本详情（如果有）
        if self._stats_invalid_records:
            lines.append(f"\n4. 无效样本详情（共{len(self._stats_invalid_records)}个）：")
            for i, err in enumerate(self._stats_invalid_records, 1):
                lines.append(f"   {i}. {err}")
        else:
            lines.append("\n4. 无效样本详情：无")

        lines.append("=" * 70)
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
    
    @staticmethod
    def convert_img_from_numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
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
    def convert_img_from_tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
        """
        将RGB格式的tensor转换为numpy格式的图像，并调整维度从(C, H, W)到(H, W, C)
        
        Args:
            img: 输入的torch.Tensor，格式为(C, H, W)
            
        Returns:
            转换后的numpy图像数组，格式为(H, W, C)
        """                
        img = img.detach().cpu()

        # tensor -> numpy
        img_np = img.permute(1,2,0).numpy()   # CHW -> HWC

        # 0~1 -> 0~255
        img_np = (img_np * 255).clip(0,255).astype(np.uint8)

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
        "/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/train.csv"
    ]  # 可以是相对路径或绝对路径
    FIELD_MAP = {
        "img_path": "data_img",  # 类内字段img对应CSV中的image_path列
        "label_path": "label_detect_yolo",  # 类内字段label对应CSV中的label_path列
    }

    dataset = BaseDataset(csv_paths=CSV_FILES, key_map=FIELD_MAP)

    dataset.cache_image(dataset.sample_path_table["img"], "cache")
    print(dataset)
