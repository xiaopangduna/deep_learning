import pytest
from pathlib import Path
from lovely_deep_learning.utils.file import list_grouped_files_from_folders

def create_file_structure(root: Path, structure: dict):
    """根据输入结构动态创建文件夹和文件"""
    for folder_name, filenames in structure.items():
        folder_path = root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (folder_path / filename).touch()


@pytest.mark.parametrize(
    "test_case_list_grouped_files_from_folders",
    [
        # 测试用例1：基础缺失场景，allow_missing=True
        {
            "name": "basic_missing_allowed",
            "input_structure": {
                "images": ["0001.jpg", "0002.jpg", "0003.jpg"],
                "labels": ["0001.txt", "0003.txt"],  # 缺少0002.txt
                "masks": ["0001.png", "0002.png", "0003.png"]
            },
            "suffix_groups": [[".jpg"], [".txt"], [".png"]],
            "allow_missing": True,
            "expected": [
                ["images/0001.jpg", "labels/0001.txt", "masks/0001.png"],
                ["images/0002.jpg", "None", "masks/0002.png"],
                ["images/0003.jpg", "labels/0003.txt", "masks/0003.png"]
            ]
        },
        # 测试用例2：基础缺失场景，allow_missing=False
        {
            "name": "basic_missing_disallowed",
            "input_structure": {
                "images": ["0001.jpg", "0002.jpg", "0003.jpg"],
                "labels": ["0001.txt", "0003.txt"],  # 缺少0002.txt
                "masks": ["0001.png", "0002.png", "0003.png"]
            },
            "suffix_groups": [[".jpg"], [".txt"], [".png"]],
            "allow_missing": False,
            "expected": [
                ["images/0001.jpg", "labels/0001.txt", "masks/0001.png"],
                ["images/0003.jpg", "labels/0003.txt", "masks/0003.png"]
            ]
        },
        # 测试用例3：多后缀支持场景（确保无同名文件）
        {
            "name": "multi_suffix_support",
            "input_structure": {
                "images": ["0001.jpg", "0002.png", "0003.jpg"],  # 无同名（不含后缀）文件
                "labels": ["0001.txt", "0002.txt", "0003.txt"],
            },
            "suffix_groups": [[".jpg", ".png"], [".txt"]],
            "allow_missing": False,
            "expected": [
                ["images/0001.jpg", "labels/0001.txt"],
                ["images/0002.png", "labels/0002.txt"],
                ["images/0003.jpg", "labels/0003.txt"]
            ]
        },
        # 测试用例4：完全匹配场景
        {
            "name": "complete_match",
            "input_structure": {
                "images": ["a.jpg", "b.jpg"],
                "labels": ["a.txt", "b.txt"],
                "masks": ["a.png", "b.png"]
            },
            "suffix_groups": [[".jpg"], [".txt"], [".png"]],
            "allow_missing": False,
            "expected": [
                ["images/a.jpg", "labels/a.txt", "masks/a.png"],
                ["images/b.jpg", "labels/b.txt", "masks/b.png"]
            ]
        },
        # 测试用例5：验证排序功能
        {
            "name": "sorted_by_stem",
            "input_structure": {
                "images": ["0003.jpg", "0001.jpg", "0002.jpg"],
                "labels": ["0003.txt", "0001.txt", "0002.txt"],
            },
            "suffix_groups": [[".jpg"], [".txt"]],
            "allow_missing": False,
            "expected": [
                ["images/0001.jpg", "labels/0001.txt"],
                ["images/0002.jpg", "labels/0002.txt"],
                ["images/0003.jpg", "labels/0003.txt"]
            ]
        }
    ],
    ids=[case["name"] for case in [
        {"name": "basic_missing_allowed"},
        {"name": "basic_missing_disallowed"},
        {"name": "multi_suffix_support"},
        {"name": "complete_match"},
        {"name": "sorted_by_stem"}
    ]]
)
def test_list_grouped_files_from_folders(test_case_list_grouped_files_from_folders, tmp_path):
    """测试list_grouped_files_from_folders函数的各种文件分组场景"""
    # 1. 根据输入结构创建文件
    create_file_structure(tmp_path, test_case_list_grouped_files_from_folders["input_structure"])
    
    # 2. 准备测试参数
    dirs = [tmp_path / folder_name for folder_name in test_case_list_grouped_files_from_folders["input_structure"].keys()]
    suffix_groups = test_case_list_grouped_files_from_folders["suffix_groups"]
    allow_missing = test_case_list_grouped_files_from_folders["allow_missing"]
    
    # 3. 执行被测试函数
    result = list_grouped_files_from_folders(
        dirs,
        suffix_groups,
        relative_to=tmp_path,
        allow_missing=allow_missing
    )
    
    # 4. 验证结果
    assert result == test_case_list_grouped_files_from_folders["expected"], (
        f"测试用例 {test_case_list_grouped_files_from_folders['name']} 失败\n"
        f"预期: {test_case_list_grouped_files_from_folders['expected']}\n"
        f"实际: {result}"
    )

    