import os
from typing import List, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


def make_inference_dataset(dir_list: List[str]):
    """
    从目录列表创建推理数据集
    
    Args:
        dir_list: 包含图像文件的目录路径列表
        
    Returns:
        List of (image_path, None) tuples
    """
    images = []
    
    for directory in dir_list:
        if not os.path.isdir(directory):
            continue
            
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append((path, None))
                    
    return images


def is_image_file(filename: str) -> bool:
    """
    检查文件是否为图像文件
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 如果是图像文件返回True
    """
    IMG_EXTENSIONS = (
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
    )
    return filename.lower().endswith(IMG_EXTENSIONS)


class ImagePredictDataset(Dataset):
    """
    用于推理的数据集类，从目录列表加载图像文件，标签设置为None
    """
    def __init__(
        self, 
        dir_list: List[str], 
        transform: Optional[Callable] = None,
        loader: Callable[[str], Image.Image] = None
    ):
        """
        Args:
            dir_list: 包含图像文件的目录路径列表
            transform: 可选的图像转换函数
            loader: 图像加载函数，默认使用PIL
        """
        self.dir_list = dir_list
        self.transform = transform
        self.loader = loader or self.default_loader
        self.samples = make_inference_dataset(dir_list)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): 索引

        Returns:
            tuple: (sample, target) where target is None for inference
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    @staticmethod
    def default_loader(path: str) -> Image.Image:
        """
        默认图像加载器
        
        Args:
            path: 图像文件路径
            
        Returns:
            PIL.Image.Image: 加载的图像
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')