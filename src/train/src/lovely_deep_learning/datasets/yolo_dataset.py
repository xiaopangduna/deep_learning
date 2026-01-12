import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from .base_dataset import BaseDataset
from typing import List, Dict, Optional, Callable, Any, Tuple, Union
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2  # 假设使用OpenCV处理图像
from copy import deepcopy
from PIL import Image, ImageOps
from torchvision import tv_tensors
import torch

class YoloDataset(BaseDataset):
    def __init__(
        self,
        # 父类必需参数（原样传递）
        csv_paths: List[str],
        key_map: Dict[str, str] = {"img_paths": "data_img", "label_paths": "label_detect_yolo"},
        transform: Optional[Callable] = None,
        # 子类特有参数（图像配置字典）
        cfgs: Optional[Dict] = {},
        cache_label_path: Optional[str] = None,
        cache_image_dir: Optional[str] = None,
    ):
        # 1. 调用父类初始化方法，完成路径解析等核心功能
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.cfgs = cfgs
        self.cache_label_path = Path(cache_label_path) if cache_label_path else None
        self.cache_image_dir = Path(cache_image_dir) if cache_image_dir else None
        self.sample_path_table["img_npy_paths"] = None  # 所有图像npy的路径，默认None
        self.samples = None  # 缓存训练需要的所有样本，包括除图像数组外从所有信息，比如，图像路径，标签路径，标签内容，原始图像尺寸，标签类别，标签框，是否归一化，标签框格式等
        # 检查是否需要缓存
        if self._should_cache_image():
            img_npy_paths = self._cache_image()
        if cache_image_dir:
            self.sample_path_table["img_npy_paths"] = img_npy_paths
        else:
            self.sample_path_table["img_npy_paths"] = [None for _ in range(len(self))]

        if self._should_cache_label():
            self._cache_label()
        if cache_label_path:
            self.samples = np.load(self.cache_label_path, allow_pickle=True).item()["samples"]
        else:
            self.samples = self._multithreaded_load_yolo_detection_samples()

    @staticmethod
    def draw_bounding_boxes(
        image: np.ndarray, 
        bboxes: np.ndarray, 
        classes: np.ndarray, 
        class_names: Optional[Dict[int, str]] = None, 
        colors: Optional[List[Tuple[int, int, int]]] = None, 
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制边界框
        
        参数:
            image: 输入图像 (numpy array)
            bboxes: 边界框 numpy 数组，格式为 [[x_center, y_center, width, height], ...] (归一化坐标)
            classes: 类别ID numpy 数组
            class_names: 类别名称字典，键为类别ID，值为类别名称 (可选)
            colors: 颜色列表，每个类别一种颜色 (可选)
            thickness: 边界框线条粗细
            
        返回:
            绘制了边界框的图像
        """
        h, w = image.shape[:2]
        
        # 如果没有提供颜色，则使用默认颜色
        if colors is None:
            colors = [(0, 255, 0)]  # BGR格式

        for i, (bbox, cls_id) in enumerate(zip(bboxes, classes)):
            # 解析边界框坐标 (x_center, y_center, width, height) - 归一化坐标
            x_center, y_center, width, height = bbox
            
            # 转换为像素坐标
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # 计算左上角和右下角坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 获取当前框的颜色
            if len(colors) > 1:
                color = colors[cls_id % len(colors)]
            else:
                color = colors[0]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 添加类别标签
            label = f"Class: {cls_id}"
            if class_names and cls_id in class_names:
                label = class_names[cls_id]
                
            # 计算标签背景框的大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            
            # 绘制标签背景
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(
                image, 
                label, 
                (x1, y1 - 5), 
                font, 
                font_scale, 
                (255, 255, 255),  # 白色文字
                1, 
                cv2.LINE_AA
            )
        
        return image

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = deepcopy(self.samples[index])
        net_in = {}
        net_out = {}
        img = read_img(sample["img_path"],sample["img_npy_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. 转为 torch.Tensor 并调整维度 (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)  # (C, H, W)
        img_tensor = img_tensor.contiguous()
        img_tv = tv_tensors.Image(img_tensor)
        # img_tv = 

        if self.transform:
            # img_tv = 
            # bboxex_tv =
            # 
            pass
            #  = self.transform()


        return sample

    def _should_cache_image(self) -> bool:
        """判断是否需要缓存图像"""

        return self.cache_image_dir is not None

    def _cache_image(self) -> List[str]:
        """缓存图像并返回缓存路径列表"""
        img_npy_paths = self.cache_image(self.sample_path_table["img_paths"], cache_dir=str(self.cache_image_dir))

        return img_npy_paths

    def _should_cache_label(self) -> bool:
        """判断是否需要缓存标签"""
        # 1. 没提供缓存路径 → 不缓存
        if self.cache_label_path is None:
            return False

        # 2. 缓存文件不存在 → 需要缓存
        if not os.path.exists(self.cache_label_path):
            return True

        # 3. 缓存存在 → 检查哈希是否一致
        img_hashes = self.get_hash(self.sample_path_table["img_paths"])
        label_hashes = self.get_hash(self.sample_path_table["label_paths"])
        cache = np.load(self.cache_label_path, allow_pickle=True).item()
        if cache["meta"]["image_hash"] != img_hashes or cache["meta"]["label_hash"] != label_hashes:
            return True

        return False

    def _cache_label(self) -> dict:
        """生成新的缓存数据"""
        # 1. 初始化全局元信息
        meta = {
            "image_hash": self.get_hash(self.sample_path_table["img_paths"]),
            "label_hash": self.get_hash(self.sample_path_table["label_paths"]),
            "sample_fields": [
                "img_path",
                "img_npy_path",
                "label_path",
                "original_shape",
                "classes",
                "bboxes",
                "normalized",
                "bbox_format",
            ],
        }
        samples = self._multithreaded_load_yolo_detection_samples()
        # 过滤损坏的样本
        # 4. 组装完整缓存并保存
        cache = {"meta": meta, "samples": samples}
        with open(str(self.cache_label_path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, cache)

        return cache

    def _multithreaded_load_yolo_detection_samples(self) -> List[Dict[str, Any]]:
        """
        多线程加载所有YOLO检测样本。

        Returns:
            List[Dict[str, Any]]: 包含所有样本信息的列表，每个样本为一个字典。
        """
        samples = []
        # 3. 多线程处理所有样本
        print(f"开始生成缓存: {self.cache_label_path}")
        # 生成标签缓存
        with ThreadPool(min(8, os.cpu_count())) as pool:
            results_label = pool.imap(
                func=lambda x: read_img_and_yolo_detection_labels(*x),
                iterable=zip(
                    self.sample_path_table["img_paths"],
                    self.sample_path_table["img_npy_paths"],
                    self.sample_path_table["label_paths"],
                ),
            )
            pbar = tqdm(results_label, total=len(self), desc="处理标签")
            for img_path, img_npy_path, label_path, img, shape_ori, cls, bboxes in pbar:
                samples.append(
                    {
                        "img_path": img_path,
                        "img_npy_path": img_npy_path,
                        "label_path": label_path,
                        "original_shape": shape_ori,
                        "classes": cls,  # n, 1
                        "bboxes": bboxes,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )

        return samples


def read_img_and_yolo_detection_labels(img_path: str, img_npy_path: str, label_path: str):
    """
    从YOLO格式的.txt文件中读取目标检测标签，并转换为NumPy数组。

    此函数假设标签文件格式正确，不包含任何校验。

    Args:
        file_path (str): YOLO标签文件的路径。

    Returns:
        np.ndarray: 一个形状为 (n, 5) 的NumPy数组，其中n是目标数量。
                    每行包含 [class_id, x_center, y_center, width, height]。
                    如果文件为空或不存在，返回一个空数组 (0, 5)。
    """
    img, shape_ori = read_img(img_path, img_npy_path)
    cls, bboxes = read_yolo_detection_labels(label_path)
    return img_path, img_npy_path, label_path, img, shape_ori, cls, bboxes


def read_img(img_path: str, img_npy_path: Union[str, None]):
    if img_npy_path:
        img = np.load(img_npy_path)
    else:
        img = cv2.imread(img_path)
    return img, img.shape


def read_yolo_detection_labels(file_path: str):
    """
    从YOLO格式的.txt文件中读取目标检测标签，并转换为NumPy数组。

    此函数假设标签文件格式正确，不包含任何校验。

    Args:
        file_path (str): YOLO标签文件的路径。

    Returns:
        np.ndarray: 一个形状为 (n, 5) 的NumPy数组，其中n是目标数量。
                    每行包含 [class_id, x_center, y_center, width, height]。
                    如果文件为空或不存在，返回一个空数组 (0, 5)。
    """
    if not file_path:
        labels = np.empty((0, 5), dtype=np.float32)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
            labels = np.array(labels, dtype=np.float32)

    cls = labels[:, 0].astype(np.int32)
    bbox = labels[:, 1:5].astype(np.float32)
    return cls, bbox
