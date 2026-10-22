import csv
import os
import hashlib
from typing import List, Dict, Optional, Callable, Any, Union, Sequence
from torch.utils.data import Dataset
from tqdm import tqdm  
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch


class BaseDataset(Dataset):
    """把一份或多份 CSV 读成一张样本表，按下标返回一行字符串字典。

    本类不读取图像。图像、标签和增强由子类在 ``__getitem__`` 中完成。

    Attributes
    ----------
    csv_paths : list[Path]
        ``expanduser().resolve()`` 之后的绝对路径。
    key_map : dict[str, str]
        列重命名。键是表内列名，值是 CSV 表头。
    transform : Callable | None
        样本变换。本类只保存，由子类调用。``None`` 表示不做变换。
    sample_path_table : pandas.DataFrame
        合并并重命名后的表。单元格为字符串，空单元格为 ``""``。
        子类可以在读取样本前整表替换，``__len__`` 跟随替换后的行数。

    Notes
    -----
    读入使用 ``dtype=str, keep_default_na=False``。表头不可重复。
    重命名后的列名若冲突，本类不合并。
    重命名前，表头以 ``path`` 开头（不区分大小写）的列：第一个非空单元格若是相对路径，
    整列按该 CSV 所在目录补成绝对路径；单元格本身已是绝对路径的保持不变。
    """

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        csv_paths
            至少一张已存在的 CSV。多张表的列名集合必须与第一张相同，列顺序以第一张为准。
        key_map
            键是读入后的列名，值是 CSV 原表头。空字典表示不重命名。
            每个值必须出现在表头中。未出现在值里的列保留原表头。
        transform
            样本变换。本类只保存，由子类在 ``__getitem__`` 中调用。
        """
        if not csv_paths:
            raise ValueError("csv_paths 不能为空")
        self.csv_paths: List[Path] = [
            Path(p).expanduser().resolve() for p in csv_paths]
        self.key_map = key_map
        self.transform = transform

        self.sample_path_table: pd.DataFrame = self._generate_sample_path_table()

    def __len__(self) -> int:
        """返回 ``sample_path_table`` 的行数。"""
        return len(self.sample_path_table)

    def __getitem__(self, index: int) -> Dict[str, str]:
        """返回该行全部列。值为字符串，缺失为空字符串。"""
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
                f"\n3. 样本数：{len(self)}",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    def _generate_sample_path_table(self) -> pd.DataFrame:
        if not self.csv_paths:
            raise ValueError("csv_paths 不能为空")

        for p in self.csv_paths:
            if not p.is_file():
                raise FileNotFoundError(f"CSV 文件不存在：{p}")

        ref_headers = self.read_csv_fieldnames(self.csv_paths[0])
        for p in self.csv_paths[1:]:
            h = self.read_csv_fieldnames(p)
            assert frozenset(ref_headers) == frozenset(h), (
                f"CSV 列名不一致（忽略顺序后与首张表须相同）：首张 {self.csv_paths[0]} 为 {sorted(frozenset(ref_headers))!r}，"
                f"{p} 为 {sorted(frozenset(h))!r}"
            )
        fieldnames = list(ref_headers)
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(
                "CSV 首行表头存在重复列名：无法与 key_map、DataFrame 列一一对应（"
                "csv.DictReader 对同名列会覆盖、pandas 可能改名或产生多列）。请先修正表头。"
                f" 当前表头：{fieldnames!r}"
            )

        for inner_field, csv_col in self.key_map.items():
            if csv_col not in fieldnames:
                raise ValueError(
                    f"key_map 中的 CSV 表头 {csv_col!r}（类内字段 {inner_field!r}）不在 CSV 表头中；"
                    f"当前表头：{fieldnames}"
                )

        rename_columns = {csv_col: inner for inner,
                          csv_col in self.key_map.items()}
        col_order = [rename_columns.get(o, o) for o in fieldnames]

        frames: List[pd.DataFrame] = []
        for csv_path in self.csv_paths:
            csv_dir = csv_path.parent
            df = pd.read_csv(csv_path, encoding="utf-8",
                             dtype=str, keep_default_na=False)
            df = df[fieldnames].copy()

            self.resolve_relative_path_columns_if_needed(
                df, csv_dir, fieldnames)

            df.rename(columns=rename_columns, inplace=True)
            df = df.reindex(columns=col_order)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=col_order)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def read_csv_fieldnames(csv_path: Path) -> tuple:
        """返回 CSV 表头，顺序与文件中一致。"""
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            return tuple(fieldnames)

    @staticmethod
    def resolve_relative_path_columns_if_needed(
        df: pd.DataFrame, csv_dir: Path, fieldnames: Sequence[str]
    ) -> None:
        """按类文档中的路径列规则，就地把相对路径补成绝对路径。"""
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

    @staticmethod
    def read_img(img_path: str, img_npy_path: Union[str, None]):
        """``img_npy_path`` 有值时读 ``.npy``，否则用 OpenCV 读 ``img_path``。返回图像和 ``shape``。"""
        if img_npy_path:
            img = np.load(img_npy_path)
        else:
            img = cv2.imread(img_path)
        return img, img.shape

    @staticmethod
    def cache_image(img_paths: List[str], cache_dir: str) -> List[str]:
        """把图像存成 ``.npy`` 缓存，返回与 ``img_paths`` 等长的缓存路径。

        文件名是图像内容 MD5 的前 16 位。已有缓存直接复用。保存的是 OpenCV 读到的 BGR 数组，不做缩放或通道转换。

        Parameters
        ----------
        img_paths
            原始图像路径。
        cache_dir
            缓存目录，不存在时创建。

        Returns
        -------
        list[str]
            与 ``img_paths`` 顺序一致的 ``.npy`` 路径。

        Raises
        ------
        FileNotFoundError
            图像路径无法读取时抛出。
        """
        os.makedirs(cache_dir, exist_ok=True)

        npy_paths = []

        for img_path in tqdm(img_paths, desc="生成图像缓存", unit="张"):
            with open(img_path, "rb") as f:
                img_content = f.read()
            img_hash = hashlib.md5(img_content).hexdigest()[:16]

            npy_filename = f"{img_hash}.npy"
            npy_path = os.path.join(cache_dir, npy_filename)

            if not os.path.exists(npy_path):
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"无法读取图像文件（路径不存在或文件损坏）：{img_path}")

                np.save(npy_path, img)

            npy_paths.append(npy_path)

        return npy_paths

    @staticmethod
    def get_hash(obj: Any) -> str:
        """返回对象的 MD5 十六进制串，供缓存校验。

        列表和元组会先排序，字典按键排序。``str`` 或 ``Path`` 指向已存在的文件时，只读前 1MB。
        """
        hash_obj = hashlib.md5()

        if isinstance(obj, pd.Series):
            obj = obj.tolist()
        if isinstance(obj, (list, tuple)):
            for item in sorted(obj):
                hash_obj.update(str(item).encode("utf-8"))
        elif isinstance(obj, dict):
            for key in sorted(obj):
                hash_obj.update(f"{key}:{obj[key]}".encode("utf-8"))
        elif isinstance(obj, (str, Path)):
            obj = Path(obj)
            if obj.exists() and obj.is_file():
                with open(obj, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        hash_obj.update(chunk)
                        break
        else:
            hash_obj.update(str(obj).encode("utf-8"))

        return hash_obj.hexdigest()

    @staticmethod
    def convert_img_from_numpy_to_tensor_uint8(img: np.ndarray) -> torch.Tensor:
        """BGR ``(H, W, C)`` uint8 数组转为 RGB ``(C, H, W)`` uint8 张量。"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        img_tensor = img_tensor.contiguous()

        return img_tensor

    @staticmethod
    def convert_img_from_tensor_to_numpy_uint8(img: torch.Tensor) -> np.ndarray:
        """``(C, H, W)`` uint8 张量转为 BGR ``(H, W, 3)`` uint8 数组。

        支持 RGB（C=3）与灰度（C=1）；灰度会扩成 3 通道便于可视化画框/写字。
        """
        img = img.detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np.astype(np.uint8)

        if img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_np

    @staticmethod
    def unwrap_tv_tensor(tensor_or_tv: Any) -> torch.Tensor:
        """取出 ``tv_tensors`` 里的底层张量；普通张量原样返回。"""
        if hasattr(tensor_or_tv, "data"):
            return tensor_or_tv.data
        return tensor_or_tv

    @staticmethod
    def prepare_net_in_sample(ni: dict[str, Any]) -> dict[str, Any]:
        """单样本 ``net_in`` → collate 后形态（写入 plain ``img``，与检测侧一致）。"""
        ni = dict(ni)
        ni["img"] = BaseDataset.unwrap_tv_tensor(ni["img_tv_transformed"])
        return ni

    @staticmethod
    def collate_net_in_tuple(
        samples: Sequence[tuple[dict[str, Any], Any]],
    ) -> tuple[dict[str, Any], ...]:
        """把一批 ``(net_in, net_out)`` 中的 ``net_in`` 收成元组，并写入 plain ``img``。"""
        net_in_tuple, _ = zip(*samples)
        return tuple(BaseDataset.prepare_net_in_sample(ni) for ni in net_in_tuple)

    @staticmethod
    def stack_batch_images(net_in: Sequence[dict[str, Any]]) -> torch.Tensor:
        """沿 batch 维堆叠 ``img``。尚未写入 ``img`` 时从 ``img_tv_transformed`` 取出。"""
        tensors = []
        for ni in net_in:
            if "img" in ni:
                tensors.append(ni["img"])
            else:
                tensors.append(
                    BaseDataset.unwrap_tv_tensor(ni["img_tv_transformed"])
                )
        return torch.stack(tensors, dim=0)

    @staticmethod
    def get_collate_fn_for_dataloader():
        """返回 DataLoader 的 ``collate_fn``，输出 ``(net_in 元组, net_out 元组)``。"""
        def collate_fn(x):
            return BaseDataset.collate_net_in_tuple(x), tuple(no for _, no in x)

        return collate_fn

