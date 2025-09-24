import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from .base_dataset import BaseDataset
from typing import List, Dict, Optional, Callable, Any,Tuple
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2  # 假设使用OpenCV处理图像
import hashlib
from PIL import Image, ImageOps

class YoloDataset(BaseDataset):
    def __init__(
        self,
        # 父类必需参数（原样传递）
        csv_paths: List[str],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
        # 子类特有参数（图像配置字典）
        cfgs: Optional[Dict] = None,
        cache_label_path: Optional[str] = None,
        cache_image_dir: Optional[str] = None,
    ):
        # 1. 调用父类初始化方法，完成路径解析等核心功能
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.cfgs = cfgs or {}  # 若未提供配置，使用空字典
        self.is_cache_label = cache_label_path is not None
        self.cache_label_path = Path(cache_label_path) if self.is_cache_label else None
        self.is_cache_image = cache_image_dir is not None
        self.cache_image_dir = Path(cache_image_dir) if self.is_cache_image else None
        # 检查是否已经缓存标签
        # 检测是否已经缓图像
        if self.is_cache_image:
            # 检测缓存文件是否存在
            # 不存在则生成
            self.sample_path_table["img_npy"] = self.cache_image(
                self.sample_path_table["img_path"], cache_dir=str(self.cache_image_dir)
            )
            pass
        if self.is_cache_label:
            # 检测缓存文件是否存在
            # 不存在则生成
            self.labels = self.cache_label()
            pass


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
        return {"img": img, "bbox": bbox, "cls": cls, "original_shape": original_shape, "img_path": img_path}

    def __len__(self) -> int:
        # 若使用缓存，返回缓存中的有效样本数
        if self.cache and self.cache_data is not None:
            return len(self.cache_data["samples"])
        return super().__len__()

    def cache_label(self):
        return self._load_or_generate_cache()

    def _get_hash(self, obj: Any) -> str:
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

    def _is_cache_valid(self, cache: dict) -> bool:
        """校验缓存是否有效"""
        # 1. 校验版本
        if cache.get("meta", {}).get("version") != "1.0":
            print(f"缓存版本不匹配（需要1.0，实际{cache.get('meta', {}).get('version')}）")
            return False

        # 2. 校验数据集哈希（基于所有CSV和样本路径）
        current_dataset_hash = self._get_hash({"csv_paths": self.csv_paths, "sample_paths": self.sample_path_table})
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
            "image_hash": self._get_hash(self.sample_path_table["img_path"]),
            "label_hash": self._get_hash(self.sample_path_table["label_path"]),
            "stats": {"total_samples": self.num_samples, "corrupt_samples": 0, "negative_samples": 0},
            "sample_fields": ["img_path", "label_path", "original_shape", "is_negative", "img_array", "bbox", "cls"],
        }

        # 2. 定义单个样本的处理函数
        def process_sample(index: int) -> Optional[dict]:
            try:
                # 获取样本路径
                paths = super().__getitem__(index)
                img_path = paths.get("img_path", "")  # 对应key_map中的图像字段
                label_path = paths.get("label_path", "")  # 对应key_map中的标签字段

                # 读取图像
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(f"图像文件不存在: {img_path}")

                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"无法读取图像: {img_path}")

                original_shape = img.shape[:2]  # (H, W)

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
                    "bbox": bbox,
                    "cls": cls,
                }

            except Exception as e:
                meta["stats"]["corrupt_samples"] += 1
                print(f"处理样本{index}出错: {str(e)}")
                return None

        # 3. 多线程处理所有样本
        print(f"开始生成缓存: {self.cache_label_path}")
        with ThreadPool(min(8, os.cpu_count())) as pool:
            samples = list(
                tqdm(pool.imap(process_sample, range(self.num_samples)), total=self.num_samples, desc="生成缓存")
            )

        # 过滤损坏的样本
        valid_samples = [s for s in samples if s is not None]
        meta["stats"]["total_samples"] = len(valid_samples)  # 更新有效样本数

        # 4. 组装完整缓存并保存
        cache = {"meta": meta, "samples": valid_samples}
        with open(str(self.cache_label_path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, cache)
        print(
            f"缓存生成完成: {self.cache_label_path} (有效样本: {len(valid_samples)}, 损坏: {meta['stats']['corrupt_samples']})"
        )

        return cache

    def _load_or_generate_cache(self) -> dict:
        """加载现有缓存（如果有效），否则生成新缓存"""
        if self.cache_label_path.exists():
            try:
                # 尝试加载缓存
                cache = np.load(self.cache_label_path, allow_pickle=True).item()

                # 校验缓存有效性
                if self._is_cache_valid(cache):
                    print(f"加载有效缓存: {self.cache_label_path}")
                    return cache
                else:
                    print("缓存无效，将重新生成")
            except Exception as e:
                print(f"缓存文件损坏 ({str(e)})，将重新生成")

        # 生成新缓存
        return self._generate_cache()
    
    def a():
        pass
    
def verify_image_label(args: Tuple) -> List:
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

                # All labels
                max_cls = 0 if single_cls else lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]
    
def load_yolo_label_from_txt(label_path: str) -> np.ndarray:
    """Load YOLO format labels from a text file.

    Args:
        label_path (str): Path to the label text file.

    Returns:
        np.ndarray: Array of shape (num_boxes, 5) where each row is [class, x, y, w, h].
    """
    with open(label_path, "r") as f:
        data = f.readlines()
    data = [x.strip().split() for x in data]
    return np.array(data, dtype=np.float32)