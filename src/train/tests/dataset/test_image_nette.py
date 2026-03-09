import pytest
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "datasets" / "IMAGENETTE" / "imagenette2-320"
TMP_DIR = PROJECT_ROOT / "tmp"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


class TestImageNette:
    def test_getitem(self):
        """
        测试无 transform 的情况：__getitem__ 应返回 (PIL.Image.Image, int)
        并打印标签和图像大小。
        """
        # 使用 ImageFolder 加载数据集
        train_dir = DATA_ROOT / "train"
        dataset = datasets.ImageFolder(root=str(train_dir), transform=None)

        img, label = dataset[0]

        assert isinstance(img, Image.Image), "图像应为 PIL Image"
        assert isinstance(label, int), "标签应为整数"

        save_path = TMP_DIR / "test_IMAGENETTE.png"
        # img.save(save_path)

        print(f"无 transform: 标签={label}, 图像尺寸={img.size}, 模式={img.mode}")

    def test_getitem_with_transform(self):
        """
        测试使用 transforms.ToTensor() 的情况：
        __getitem__ 应返回 (torch.Tensor, int)
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dir = DATA_ROOT / "train"
        dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)

        img_tensor, label = dataset[0]

        assert isinstance(img_tensor, torch.Tensor), "图像应为 torch.Tensor"
        assert isinstance(label, int), "标签应为整数"
        assert img_tensor.shape == (3, 224, 224), "张量形状应为 (3, 224, 224)"
        assert img_tensor.dtype == torch.float32, "张量数据类型应为 torch.float32"

        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)
        save_path = TMP_DIR / "test_IMAGENETTE_with_transform.png"
        # img_pil.save(save_path)
        assert save_path.exists(), "图像文件应成功保存"

        print(f"有 transform: 标签={label}, 图像保存至 {save_path}")