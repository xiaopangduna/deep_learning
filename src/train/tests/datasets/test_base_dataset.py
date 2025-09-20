import pytest
import csv
import os
from pathlib import Path
from torch.utils.data import DataLoader
from lovely_deep_learning.datasets.base_dataset import BaseDataset  # 替换为实际模块路径


def create_test_csv(csv_path: Path, headers: list, rows: list):
    """创建测试CSV文件，写入相对路径"""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


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