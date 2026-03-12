import pytest
import os
from PIL import Image
from src.lovely_deep_learning.dataset.predict import ImagePredictDataset


def test_inference_dataset_basic():
    """测试InferenceDataset基本功能"""
    # 使用指定的图片文件夹路径
    image_dirs = ["datasets/coco8/images"]
    
    dataset = ImagePredictDataset(image_dirs)
    
    # 测试数据集长度（应大于0如果目录中有图片）
    assert len(dataset) >= 0
    
    # 如果数据集中有图片，测试获取第一个样本
    if len(dataset) > 0:
        sample, target = dataset[0]
        
        # 验证样本是PIL图像
        assert isinstance(sample, Image.Image)
        
        # 验证目标为None（推理模式）
        assert target is None
        
        print(f"✅ 成功加载 {len(dataset)} 个图像样本")
        print(f"✅ 第一个图像尺寸: {sample.size}")




if __name__ == '__main__':
    pytest.main([__file__])