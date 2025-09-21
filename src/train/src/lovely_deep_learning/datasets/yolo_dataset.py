import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from .base_dataset import BaseDataset
from typing import List, Dict, Optional, Callable, Any
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2  # 假设使用OpenCV处理图像
import hashlib


class YoloDataset(BaseDataset):
    def __init__(
        self,
        # 父类必需参数（原样传递）
        csv_paths: List[str],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
        # 子类特有参数（图像配置字典）
        cfgs: Optional[Dict] = None,
        cache: bool = True,
    ):
        # 1. 调用父类初始化方法，完成路径解析等核心功能
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.cache = cache
        self.cache_path = self._get_default_cache_path()  # 自动生成缓存路径
        self.cache_data = None  # 存储完整缓存数据（包含meta和samples）
        # 2. 处理子类特有配置（默认值+参数校验）
        self.cfgs = cfgs or {}  # 若未提供配置，使用空字典

        # 3.验证参数合理性


    # def __getitem__(self, index):
    #     # 获取路径
    #     sample_container = super().__getitem__(index)

    #     return sample_container
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"索引超出范围: {index} (总样本数: {len(self)})")

        # 1. 从缓存读取或实时加载
        if self.cache and self.cache_data is not None:
            # 从缓存获取数据
            sample = self.cache_data["samples"][index]
            img = sample["img_array"].copy()  # 防止数组被意外修改
            bbox = sample["bbox"].copy()
            cls = sample["cls"].copy()
            original_shape = sample["original_shape"]
            img_path = sample["img_path"]
        else:
            # 实时加载和预处理
            paths = super().__getitem__(index)
            img_path = paths.get("img", "")
            label_path = paths.get("label", "")

            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_shape = img.shape[:2]
            
            # 预处理
            img = cv2.resize(img, (self.cfgs["img_size"], self.cfgs["img_size"]))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

            # 解析标签
            bbox = np.array([], dtype=np.float32)
            cls = np.array([], dtype=np.int32)
            
            if label_path and os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if lines:
                    labels = np.array([list(map(float, line.split())) for line in lines])
                    cls = labels[:, 0].astype(np.int32)
                    bbox = labels[:, 1:5].astype(np.float32)

        # 2. 应用动态数据增强
        if self.transform is not None:
            # 假设transform函数接受(img, bbox, cls)并返回处理后的数据
            img, bbox, cls = self.transform(img, bbox, cls)

        # 3. 返回最终数据
        return {
            "img": img,
            "bbox": bbox,
            "cls": cls,
            "original_shape": original_shape,
            "img_path": img_path
        }

    def __len__(self) -> int:
        # 若使用缓存，返回缓存中的有效样本数
        if self.cache and self.cache_data is not None:
            return len(self.cache_data["samples"])
        return super().__len__()
    
    def _get_default_cache_path(self) -> Path:
        """根据csv_paths生成缓存路径：单文件用文件名，多文件用{第一个文件名}_combined"""
        if not self.csv_paths:
            raise ValueError("csv_paths不能为空，无法生成缓存路径")

        first_csv = Path(self.csv_paths[0])
        csv_dir = first_csv.parent
        first_csv_stem = first_csv.stem

        # 单CSV文件 vs 多CSV文件
        if len(self.csv_paths) == 1:
            cache_filename = f"{first_csv_stem}.cache"
        else:
            cache_filename = f"{first_csv_stem}_combined.cache"

        return csv_dir / cache_filename
    
    def _get_hash(self, obj: Any) -> str:
        """生成任意对象的MD5哈希值，用于缓存校验"""
        hash_obj = hashlib.md5()
        
        if isinstance(obj, (list, tuple)):
            # 对列表/元组，排序后序列化（确保顺序不影响哈希）
            for item in sorted(obj):
                hash_obj.update(str(item).encode('utf-8'))
        elif isinstance(obj, dict):
            # 对字典，按键排序后序列化
            for key in sorted(obj):
                hash_obj.update(f"{key}:{obj[key]}".encode('utf-8'))
        elif isinstance(obj, (str, Path)):
            # 对文件路径，读取文件内容生成哈希
            obj = Path(obj)
            if obj.exists() and obj.is_file():
                # 读取前1MB内容（平衡效率和准确性）
                with open(obj, 'rb') as f:
                    while chunk := f.read(1024 * 1024):
                        hash_obj.update(chunk)
                        break  # 仅读1MB
        else:
            # 其他类型直接序列化
            hash_obj.update(str(obj).encode('utf-8'))
            
        return hash_obj.hexdigest()
    
    def _is_cache_valid(self, cache: dict) -> bool:
        """校验缓存是否有效"""
        # 1. 校验版本
        if cache.get("meta", {}).get("version") != "1.0":
            print(f"缓存版本不匹配（需要1.0，实际{cache.get('meta', {}).get('version')}）")
            return False

        # 2. 校验数据集哈希（基于所有CSV和样本路径）
        current_dataset_hash = self._get_hash({
            "csv_paths": self.csv_paths,
            "sample_paths": self.sample_path_table
        })
        if cache.get("meta", {}).get("dataset_hash") != current_dataset_hash:
            print("数据集已修改，缓存无效")
            return False

        # 3. 校验预处理配置哈希
        current_preprocess_hash = self._get_hash(self.cfgs)
        if cache.get("meta", {}).get("preprocess_hash") != current_preprocess_hash:
            print("预处理配置已修改，缓存无效")
            return False

        # 4. 校验样本字段完整性
        expected_fields = set(cache.get("meta", {}).get("sample_fields", []))
        if not expected_fields:
            print("缓存缺少样本字段定义")
            return False
            
        if not set(cache["samples"][0].keys()) == expected_fields:
            print("样本字段不匹配，缓存无效")
            return False

        return True
    
    def _generate_cache(self) -> dict:
        """生成新的缓存数据"""
        # 1. 初始化全局元信息
        meta = {
            "version": "1.0",
            "dataset_hash": self._get_hash({
                "csv_paths": self.csv_paths,
                "sample_paths": self.sample_path_table
            }),
            "preprocess_hash": self._get_hash(self.cfgs),
            "stats": {
                "total_samples": self.num_samples,
                "corrupt_samples": 0,
                "negative_samples": 0
            },
            "sample_fields": [
                "img_path", "label_path", "original_shape", 
                "is_negative", "img_array", "bbox", "cls"
            ]
        }

        # 2. 定义单个样本的处理函数
        def process_sample(index: int) -> Optional[dict]:
            try:
                # 获取样本路径
                paths = super().__getitem__(index)
                img_path = paths.get("img", "")  # 对应key_map中的图像字段
                label_path = paths.get("label", "")  # 对应key_map中的标签字段

                # 读取图像
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(f"图像文件不存在: {img_path}")
                
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"无法读取图像: {img_path}")
                
                original_shape = img.shape[:2]  # (H, W)

                # 预处理图像（缩放、归一化等）
                img_resized = cv2.resize(img, (self.cfgs["img_size"], self.cfgs["img_size"]))
                img_array = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # BGR转RGB
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                img_array = img_array.astype(np.float32) / 255.0  # 归一化到[0,1]

                # 解析标签
                bbox = np.array([], dtype=np.float32)
                cls = np.array([], dtype=np.int32)
                is_negative = True

                if label_path and os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    
                    if lines:
                        # 假设标签格式: 类别 x_center y_center width height (归一化)
                        labels = np.array([list(map(float, line.split())) for line in lines])
                        cls = labels[:, 0].astype(np.int32)
                        bbox = labels[:, 1:5].astype(np.float32)  # (n, 4)
                        is_negative = False
                    else:
                        meta["stats"]["negative_samples"] += 1
                else:
                    meta["stats"]["negative_samples"] += 1

                # 返回样本数据（平级结构，无meta子键）
                return {
                    "img_path": img_path,
                    "label_path": label_path,
                    "original_shape": original_shape,
                    "is_negative": is_negative,
                    "img_array": img_array,
                    "bbox": bbox,
                    "cls": cls
                }

            except Exception as e:
                meta["stats"]["corrupt_samples"] += 1
                print(f"处理样本{index}出错: {str(e)}")
                return None

        # 3. 多线程处理所有样本
        print(f"开始生成缓存: {self.cache_path}")
        with ThreadPool(min(8, os.cpu_count())) as pool:
            samples = list(tqdm(
                pool.imap(process_sample, range(self.num_samples)),
                total=self.num_samples,
                desc="生成缓存"
            ))

        # 过滤损坏的样本
        valid_samples = [s for s in samples if s is not None]
        meta["stats"]["total_samples"] = len(valid_samples)  # 更新有效样本数

        # 4. 组装完整缓存并保存
        cache = {
            "meta": meta,
            "samples": valid_samples
        }
        np.save(self.cache_path, cache)
        print(f"缓存生成完成: {self.cache_path} (有效样本: {len(valid_samples)}, 损坏: {meta['stats']['corrupt_samples']})")
        
        return cache

    def _load_or_generate_cache(self) -> dict:
        """加载现有缓存（如果有效），否则生成新缓存"""
        if self.cache_path.exists():
            try:
                # 尝试加载缓存
                cache = np.load(self.cache_path, allow_pickle=True).item()
                
                # 校验缓存有效性
                if self._is_cache_valid(cache):
                    print(f"加载有效缓存: {self.cache_path}")
                    return cache
                else:
                    print("缓存无效，将重新生成")
            except Exception as e:
                print(f"缓存文件损坏 ({str(e)})，将重新生成")

        # 生成新缓存
        return self._generate_cache()