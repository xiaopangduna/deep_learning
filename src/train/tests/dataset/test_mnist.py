import pytest

from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets" / "MNIST"
TMP_DIR = PROJECT_ROOT / "tmp"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


class TestMNIST:
    def test_getitem(self):
        """
        测试无 transform 的情况：__getitem__ 应返回 (PIL.Image.Image, int)
        并保存图像到临时文件。
        """
        # 加载数据集（如果本地没有则自动下载）
        dataset = datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=None
        )

        img, label = dataset[0]

        assert isinstance(img, Image.Image), "图像应为 PIL Image"
        assert isinstance(label, int), "标签应为整数"

        assert img.size == (28, 28), "图像尺寸应为 28x28"
        assert img.mode == "L", "图像模式应为灰度（L）"

        # 保存图像到临时目录
        save_path = TMP_DIR / "test_MNIST.png"
        # img.save(save_path)

        print(f"无 transform: 标签={label}, 图像保存至 {save_path}")

    def test_getitem_with_transform(self):
        """
        测试使用 transforms.ToTensor() 的情况：
        __getitem__ 应返回 (torch.Tensor, int)
        并保存图像到文件。
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset = datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=transform
        )

        img_tensor, label = dataset[0]

        assert isinstance(img_tensor, torch.Tensor), "图像应为 torch.Tensor"
        assert isinstance(label, int), "标签应为整数"

        # 断言张量属性
        assert img_tensor.shape == (1, 28, 28), "张量形状应为 (1, 28, 28)"
        assert img_tensor.dtype == torch.float32, "张量数据类型应为 torch.float32"

        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)
        save_path = TMP_DIR / "test_MNIST_with_transform.png"
        img_pil.save(save_path)
        assert save_path.exists(), "图像文件应成功保存"

        print(f"有 transform: 标签={label}, 图像保存至 {save_path}")
