import pytest
import csv
import os
import numpy as np
import shutil
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from lovely_deep_learning.datasets.base_dataset import BaseDataset  # 替换为实际模块路径


def create_test_csv(csv_path: Path, headers: list, rows: list):
    """创建测试CSV文件，写入相对路径"""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def create_test_image(img_path: Path, color: tuple = (0, 0, 0)):
    """创建测试图像（BGR格式，与OpenCV读取一致默认默认一致）"""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:, :, 0] = color[0]  # B通道
    img[:, :, 1] = color[1]  # G通道
    img[:, :, 2] = color[2]  # R通道
    cv2.imwrite(str(img_path), img)


@pytest.mark.parametrize(
    "test_case",
    [
        # 测试用例1：基础正常场景（所有路径有效）
        {
            "name": "basic_valid_paths",
            "csv_headers": ["data_img", "label_detect_yolo"],
            "csv_rows": [
                {"data_img": "images/001.jpg", "label_detect_yolo": "labels/001.txt"},
                {"data_img": "images/002.jpg", "label_detect_yolo": "labels/002.txt"},
                {"data_img": "images/003.jpg", "label_detect_yolo": "labels/003.txt"}
            ],
            "file_structure": {
                "images": ["001.jpg", "002.jpg", "003.jpg"],
                "labels": ["001.txt", "002.txt", "003.txt"]
            },
            "key_map": {"img": "data_img", "label": "label_detect_yolo"},
            "expected_num_samples": 3,
            "expected_invalid_count": 0  # 无效样本数=0
        },
        # 测试用例2：包含无效路径（2个无效样本）
        {
            "name": "with_invalid_paths",
            "csv_headers": ["data_img", "label_detect_yolo"],
            "csv_rows": [
                {"data_img": "images/001.jpg", "label_detect_yolo": "labels/001.txt"},  # 有效
                {"data_img": "images/002.jpg", "label_detect_yolo": "labels/002.txt"},  # 无效（图片缺失）
                {"data_img": "images/003.jpg", "label_detect_yolo": "labels/003.txt"}   # 无效（标签缺失）
            ],
            "file_structure": {
                "images": ["001.jpg", "003.jpg"],  # 缺少002.jpg
                "labels": ["001.txt", "002.txt"]   # 缺少003.txt
            },
            "key_map": {"img": "data_img", "label": "label_detect_yolo"},
            "expected_num_samples": 1,
            "expected_invalid_count": 2  # 无效样本数=2（2条记录）
        },
        # 测试用例3：多CSV文件合并
        {
            "name": "multiple_csv_files",
            "csv_files": ["train.csv", "val.csv"],
            "csv_headers": ["data_img", "label_detect_yolo"],
            "csv_rows": [
                [{"data_img": "images/001.jpg", "label_detect_yolo": "labels/001.txt"}],
                [{"data_img": "images/002.jpg", "label_detect_yolo": "labels/002.txt"}]
            ],
            "file_structure": {
                "images": ["001.jpg", "002.jpg"],
                "labels": ["001.txt", "002.txt"]
            },
            "key_map": {"img": "data_img", "label": "label_detect_yolo"},
            "expected_num_samples": 2,
            "expected_invalid_count": 0
        },
        # 测试用例4：多字段映射
        {
            "name": "multi_field_mapping",
            "csv_headers": ["data_img", "label_txt", "mask_png"],
            "csv_rows": [
                {"data_img": "images/001.jpg", "label_txt": "labels/001.txt", "mask_png": "masks/001.png"}
            ],
            "file_structure": {
                "images": ["001.jpg"],
                "labels": ["001.txt"],
                "masks": ["001.png"]
            },
            "key_map": {"img": "data_img", "label": "label_txt", "mask": "mask_png"},
            "expected_num_samples": 1,
            "expected_invalid_count": 0
        },
        # 测试用例5：CSV缺失必需字段（抛出错误）
        {
            "name": "missing_csv_fields",
            "csv_headers": ["data_img"],
            "csv_rows": [{"data_img": "images/001.jpg"}],
            "file_structure": {"images": ["001.jpg"]},
            "key_map": {"img": "data_img", "label": "label_detect_yolo"},
            "should_raise": True,
            "error_type": ValueError
        }
    ],
    ids=[case["name"] for case in [
        {"name": "basic_valid_paths"},
        {"name": "with_invalid_paths"},
        {"name": "multiple_csv_files"},
        {"name": "multi_field_mapping"},
        {"name": "missing_csv_fields"}
    ]]
)
def test_base_dataset_initialization(test_case, tmp_path):
    """测试BaseDataset的路径解析和样本加载"""
    # 1. 创建文件结构
    for folder, files in test_case["file_structure"].items():
        folder_path = tmp_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            (folder_path / file).touch()

    # 2. 创建CSV文件（与数据文件夹同级）
    csv_paths = []
    if "csv_files" in test_case:
        for i, fname in enumerate(test_case["csv_files"]):
            csv_path = tmp_path / fname
            create_test_csv(csv_path, test_case["csv_headers"], test_case["csv_rows"][i])
            csv_paths.append(str(csv_path))
    else:
        csv_path = tmp_path / "data.csv"
        create_test_csv(csv_path, test_case["csv_headers"], test_case["csv_rows"])
        csv_paths.append(str(csv_path))

    # 3. 执行测试
    if test_case.get("should_raise", False):
        with pytest.raises(test_case["error_type"]):
            BaseDataset(csv_paths=csv_paths, key_map=test_case["key_map"])
    else:
        # 初始化数据集并打印调试信息
        dataset = BaseDataset(csv_paths=csv_paths, key_map=test_case["key_map"])
        # dataset.print_statistics(detailed=True)  # 调试时打开，查看路径解析详情
        
        # 3.1 验证有效样本数
        assert len(dataset) == test_case["expected_num_samples"], \
            f"有效样本数不匹配: 预期{test_case['expected_num_samples']}, 实际{len(dataset)}"
        
        # 3.2 验证无效样本数（核心修复：直接取记录数）
        invalid_sample_count = len(dataset._stats_invalid_records)
        assert invalid_sample_count == test_case["expected_invalid_count"], \
            f"无效样本数不匹配: 预期{test_case['expected_invalid_count']}, 实际{invalid_sample_count}"
        
        # 3.3 验证解析后的路径存在
        if test_case["expected_num_samples"] > 0:
            first_sample = dataset[0]
            for field, path in first_sample.items():
                assert os.path.exists(path), \
                    f"解析后的路径不存在: {path} (字段: {field})"
                
        # 3.4 验证DataLoader兼容性
        dataloader = DataLoader(dataset, batch_size=2)
        assert len(list(dataloader)) >= 1, "DataLoader无法加载到有效样本"

# -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        # 测试用例1：基础正常场景（生成新缓存，保留BGR格式）
        {
            "name": "basic_cache_generation",
            "file_structure": {
                "images": ["001.jpg", "002.jpg", "003.jpg"]
            },
            "image_colors": [
                (0, 0, 255),   # BGR红色（显示为红色）
                (0, 255, 0),   # BGR绿色
                (255, 0, 0)    # BGR蓝色
            ],
            "expected_cache_count": 3,
            "reuse_cache": False
        },
        # 测试用例2：缓存复用场景（已存在的缓存不重新生成）
        {
            "name": "cache_reuse",
            "file_structure": {
                "images": ["001.jpg", "002.jpg"]
            },
            "image_colors": [
                (100, 100, 100),  # BGR灰色
                (200, 200, 200)   # BGR浅灰色
            ],
            "expected_cache_count": 2,
            "reuse_cache": True
        },
        # 测试用例3：相同内容图像复用缓存
        {
            "name": "duplicate_image_cache",
            "file_structure": {
                "images": ["001.jpg", "001_dup.jpg", "001_copy.jpg"]
            },
            "image_colors": [
                (0, 255, 255),  # BGR黄色（3张图内容相同）
                (0, 255, 255),
                (0, 255, 255)
            ],
            "expected_cache_count": 1,  # 3张相同图应共用1个缓存
            "reuse_cache": False
        },
        # 测试用例4：包含无效图像路径（应抛出异常）
        {
            "name": "invalid_image_path",
            "file_structure": {
                "images": ["001.jpg"]  # 实际只存在001.jpg
            },
            "image_colors": [(0, 0, 0)],  # BGR黑色
            "used_images": ["001.jpg", "002.jpg"],  # 002.jpg不存在
            "should_raise": True,
            "error_type": FileNotFoundError
        }
    ],
    ids=[case["name"] for case in [
        {"name": "basic_cache_generation"},
        {"name": "cache_reuse"},
        {"name": "duplicate_image_cache"},
        {"name": "invalid_image_path"}
    ]]
)
def test_base_dataset_cache_image(test_case, tmp_path):
    """测试BaseDataset.cache_image静态方法（适配无通道转换的逻辑）"""
    # 1. 创建文件结构
    for folder, files in test_case["file_structure"].items():
        folder_path = tmp_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        # 创建图像文件（按指定BGR颜色）
        for i, file in enumerate(files):
            img_path = folder_path / file
            color = test_case["image_colors"][i % len(test_case["image_colors"])]
            create_test_image(img_path, color)

    # 2. 准备图像路径列表
    img_folder = tmp_path / "images"
    if "used_images" in test_case:
        img_paths = [str(img_folder / fname) for fname in test_case["used_images"]]
    else:
        img_paths = [str(img_folder / fname) for fname in test_case["file_structure"]["images"]]

    # 3. 准备缓存目录
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)

    # 4. 预先生成缓存（用于复用场景）
    if test_case.get("reuse_cache", False):
        BaseDataset.cache_image(img_paths=img_paths, cache_dir=str(cache_dir))
        # 记录初始缓存文件修改时间
        initial_mtimes = {
            fname: os.path.getmtime(str(cache_dir / fname))
            for fname in os.listdir(cache_dir)
        }

    # 5. 执行测试
    if test_case.get("should_raise", False):
        # 测试异常场景
        with pytest.raises(test_case["error_type"]):
            BaseDataset.cache_image(img_paths=img_paths, cache_dir=str(cache_dir))
    else:
        # 测试正常场景
        npy_paths = BaseDataset.cache_image(img_paths=img_paths, cache_dir=str(cache_dir))
        
        # 6. 验证缓存路径数量
        assert len(npy_paths) == len(img_paths), \
            f"缓存路径数量不匹配: 预期{len(img_paths)}, 实际{len(npy_paths)}"
        
        # 7. 验证缓存文件数量
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == test_case["expected_cache_count"], \
            f"缓存文件数量不匹配: 预期{test_case['expected_cache_count']}, 实际{len(cache_files)}"
        
        # 8. 验证缓存内容正确性（**核心修改**：直接对比BGR格式，不做通道转换）
        for img_path, npy_path in zip(img_paths, npy_paths):
            # 读取原始图像（OpenCV默认返回BGR）
            orig_img = cv2.imread(img_path)  # 此时orig_img是BGR格式
            
            # 读取缓存图像（与原始图像一样是BGR格式）
            cached_img = np.load(npy_path)
            
            # 直接对比BGR内容（不再转换为RGB）
            assert np.array_equal(cached_img, orig_img), \
                f"缓存内容与原始图像不符: {img_path}\n原始图像形状: {orig_img.shape}, 缓存形状: {cached_img.shape}"
        
        # 9. 验证缓存复用（修改时间不变）
        if test_case.get("reuse_cache", False):
            for fname in cache_files:
                current_mtime = os.path.getmtime(str(cache_dir / fname))
                assert current_mtime == initial_mtimes[fname], \
                    f"缓存文件被重复生成: {fname}"
