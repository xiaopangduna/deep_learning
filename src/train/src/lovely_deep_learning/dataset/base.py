import csv
import os
import hashlib
from typing import List, Dict, Optional, Callable, Any, Union, Sequence
from torch.utils.data import Dataset
from tqdm import tqdm  # 直接引入tqdm用于进度显示
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch


class BaseDataset(Dataset):
    """
    基于 CSV 的 PyTorch Dataset：将一张或多张表读入为 ``sample_path_table``（pandas DataFrame），
    按索引取出整行字典供训练代码使用；另提供若干与图像/缓存相关的静态工具方法。

    主要属性
    --------
    csv_paths : List[Path]
        构造时传入的路径经 ``expanduser().resolve()`` 后的绝对路径列表。
    key_map : Dict[str, str]
        列重映射：键为输出列名（类内字段名），值为 CSV 原始表头名。
    transform : Optional[Callable]
        预留的数据增强/变换，本类 ``__getitem__`` 中未调用，由子类使用。
    sample_path_table : pd.DataFrame
        合并后的样本表；列名为重映射后的名字，行为全部数据行（多 CSV 纵向拼接）。
    num_samples : int
        行数，与 ``len(dataset)`` 一致。

    CSV 与 key_map 约定
    -------------------
    - ``csv_paths`` 非空；每个元素须为已存在的 CSV 文件。
    - 首张 CSV 的第一行决定列名及列顺序；其余文件的列名集合须与首张相同（顺序可不同），
      不一致时在步骤 2 中 ``assert`` 失败（使用 ``python -O`` 时断言会被关闭，不建议在生产依赖）。
    - 首行表头不得重复；否则无法与列一一对应。
    - ``key_map`` 中每一个「值」必须出现在表头中；未出现在 ``key_map`` 值里的列保留原表头名。
    - 重命名规则：仅 ``DataFrame.rename``，即 CSV 列名 → ``key_map`` 的键；若重命名后列名冲突，本类不处理。
    - 读入使用 ``pd.read_csv(..., dtype=str, keep_default_na=False)``，空单元格一般为 ``""``。

    路径列（表头以 ``path`` 开头，不区分大小写）
    -------------------------------------------
    仅针对**重命名前**的 CSV 列名判断。对每一列：若第一个非空单元为相对路径，则该列整列按
    **当前 CSV 文件所在目录**补全为绝对路径；若首个非空为绝对路径或列全空，则该列保持 read_csv 结果不变。
    列内若混有相对/绝对，在「整列参与补全」的前提下仍对每格单独判断 ``os.path.isabs``。

    Dataset 接口
    ------------
    - ``__len__``：返回 ``num_samples``。
    - ``__getitem__(i)``：返回第 ``i`` 行所有列组成的 ``Dict[str, str]``（含未出现在 ``key_map`` 中的列）。

    静态工具方法（节选）
    --------------------
    ``cache_image``、``get_hash``、``read_img``、``convert_img_*_uint8`` 等，供子类或训练流程复用，
    与 CSV 加载逻辑相互独立。
    """

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
    ):
        if not csv_paths:
            raise ValueError("csv_paths 不能为空")
        self.csv_paths: List[Path] = [
            Path(p).expanduser().resolve() for p in csv_paths]
        self.key_map = key_map
        self.transform = transform

        self.sample_path_table: pd.DataFrame = pd.DataFrame()
        self.num_samples: int = 0

        self.sample_path_table = self._generate_sample_path_table()
        self.num_samples = self._count_samples()

    def _generate_sample_path_table(self) -> pd.DataFrame:
        # 1. 校验路径列表非空，且每个 CSV 文件存在。
        # 1.1 无路径则无法确定读哪些表。
        if not self.csv_paths:
            raise ValueError("csv_paths 不能为空")

        # 1.2 路径须解析为已存在的普通文件（非目录）。
        for p in self.csv_paths:
            if not p.is_file():
                raise FileNotFoundError(f"CSV 文件不存在：{p}")

        # 2. 以首张 CSV 的表头顺序为基准；其余文件的列名集合须与首张一致（顺序可不同）。
        # 2.1 首张表第一行即列名，顺序贯穿后续 df[fieldnames] / col_order。
        ref_headers = self.read_csv_fieldnames(self.csv_paths[0])
        # 2.2 其余文件只比「列名集合」，顺序可与首张不同。
        for p in self.csv_paths[1:]:
            h = self.read_csv_fieldnames(p)
            assert frozenset(ref_headers) == frozenset(h), (
                f"CSV 列名不一致（忽略顺序后与首张表须相同）：首张 {self.csv_paths[0]} 为 {sorted(frozenset(ref_headers))!r}，"
                f"{p} 为 {sorted(frozenset(h))!r}"
            )
        # 2.3 统一成 list；禁止首行重复列名，否则与 key_map / DataFrame 无法一一对应。
        fieldnames = list(ref_headers)
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(
                "CSV 首行表头存在重复列名：无法与 key_map、DataFrame 列一一对应（"
                "csv.DictReader 对同名列会覆盖、pandas 可能改名或产生多列）。请先修正表头。"
                f" 当前表头：{fieldnames!r}"
            )

        # 3. 校验 key_map 中每个「值」均为合法表头名。
        # 3.1 「值」= CSV 原始列名，必须出现在 fieldnames 中，否则无法重命名。
        for inner_field, csv_col in self.key_map.items():
            if csv_col not in fieldnames:
                raise ValueError(
                    f"key_map 中的 CSV 表头 {csv_col!r}（类内字段 {inner_field!r}）不在 CSV 表头中；"
                    f"当前表头：{fieldnames}"
                )

        # 4. 仅按 key_map 做列重命名（值→键），其余列名不变；输出列顺序与 fieldnames 一致（映射后）。
        # 4.1 pandas.rename 用：旧列名（CSV）-> 新列名（类内字段）。
        rename_columns = {csv_col: inner for inner,
                          csv_col in self.key_map.items()}
        # 4.2 按首张表列顺序，写出重命名后的列名列表（未映射的列保持原名）。
        col_order = [rename_columns.get(o, o) for o in fieldnames]

        # 5. 逐文件 read_csv、按 fieldnames 对齐；path 前缀列按需补全相对路径；重命名、按 col_order 排布后纵向 concat。
        frames: List[pd.DataFrame] = []
        for csv_path in self.csv_paths:
            csv_dir = csv_path.parent
            # 5.1 全文本读入，避免空单元变 NaN，便于与 strip / 路径判断一致。
            df = pd.read_csv(csv_path, encoding="utf-8",
                             dtype=str, keep_default_na=False)
            # 5.2 按首张表列顺序取列（步骤 2 已用 csv 校验各文件列名集合一致，与 read_csv 表头一致）。
            df = df[fieldnames].copy()

            # 5.3 表头以 path 开头的列：若首行非空样本为相对路径，则整列按本 CSV 所在目录补全为绝对路径。
            self.resolve_relative_path_columns_if_needed(
                df, csv_dir, fieldnames)

            # 5.4 应用 key_map 列重命名。
            df.rename(columns=rename_columns, inplace=True)
            # 5.5 列顺序与 col_order 一致（与 4.2 相同语义）。
            df = df.reindex(columns=col_order)
            frames.append(df)

        # 5.6 无任何文件时返回空表结构；否则纵向合并所有行。
        if not frames:
            return pd.DataFrame(columns=col_order)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def read_csv_fieldnames(csv_path: Path) -> tuple:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            return tuple(fieldnames)

    @staticmethod
    def resolve_relative_path_columns_if_needed(
        df: pd.DataFrame, csv_dir: Path, fieldnames: Sequence[str]
    ) -> None:
        """
        就地处理 df：对 fieldnames 中「表头以 path 开头」且存在于 df 的列，
        若该列第一个非空单元格为相对路径，则整列按 csv_dir 补全为绝对路径；
        若首个非空为绝对路径或列全空，则不修改该列。
        """
        for col in fieldnames:
            if col not in df.columns:
                continue
            if not col.lower().startswith("path"):
                continue
            s = df[col].fillna("").astype(str)
            nonempty = s[s.str.strip() != ""]
            if nonempty.empty:
                continue
            first = nonempty.iloc[0].strip()
            if os.path.isabs(first):
                continue

            def abs_one(v: Any) -> str:
                t = str(v).strip() if v is not None else ""
                if t == "" or t.lower() == "nan":
                    return ""
                if os.path.isabs(t):
                    return t
                return str((csv_dir / t).resolve())

            df[col] = df[col].apply(abs_one)

    def _count_samples(self) -> int:
        if self.sample_path_table.empty:
            return 0
        return len(self.sample_path_table)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, str]:
        row = self.sample_path_table.iloc[index]
        out: Dict[str, str] = {}
        for col in self.sample_path_table.columns:
            v = row[col]
            if pd.isna(v):
                out[str(col)] = ""
            else:
                out[str(col)] = str(v)
        return out

    def __str__(self) -> str:
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

        if isinstance(obj, pd.Series):
            obj = obj.tolist()
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

        return collate_fn

