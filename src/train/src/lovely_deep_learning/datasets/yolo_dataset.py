import cv2
import numpy as np
from typing import List, Dict, Optional, Callable

from .base_dataset import BaseDataset


class YoloDataset(BaseDataset):
    def __init__(
        self,
        # 父类必需参数（原样传递）
        csv_paths: List[str],
        key_map: Dict[str, str],
        transform: Optional[Callable] = None,
        # 子类特有参数（图像配置字典）
        cfgs: Optional[Dict] = None,
    ):
        # 1. 调用父类初始化方法，完成路径解析等核心功能
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)

        # 2. 处理子类特有配置（默认值+参数校验）
        self.cfgs = cfgs or {}  # 若未提供配置，使用空字典

        # 3.验证参数合理性


    def __getitem__(self, index):
        # 获取路径
        sample_container = super().__getitem__(index)

        return sample_container
