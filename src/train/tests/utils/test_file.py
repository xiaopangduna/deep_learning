import pytest
from pathlib import Path
from lovely_deep_learning.utils.file import list_grouped_files_from_folders, split_list_by_ratio

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

    
@pytest.mark.parametrize(
    "test_case_split_list_by_ratio",
    [
        # 测试用例1：基础2:8拆分（无打乱）
        {
            "name": "basic_2_8_split_no_shuffle",
            "input_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10个元素
            "ratios": [0.2, 0.8],
            "shuffle": False,
            "expected_lengths": [2, 8],  # 按比例应拆分出2和8个元素
            "expected_ordered": True  # 未打乱，应保持原顺序
        },
        # 测试用例2：基础3:3:4拆分（有打乱）
        {
            "name": "basic_3_3_4_split_with_shuffle",
            "input_list": list(range(10)),  # 0-9共10个元素
            "ratios": [0.3, 0.3, 0.4],
            "shuffle": True,
            "expected_lengths": [3, 3, 4],
            "expected_ordered": False  # 打乱后，不应保持原顺序
        },
        # 测试用例3：单比例1.0（不拆分）
        {
            "name": "single_ratio_no_split",
            "input_list": ["a", "b", "c"],
            "ratios": [1.0],
            "shuffle": False,
            "expected_lengths": [3],
            "expected_ordered": True
        },
        # 测试用例4：元素数量少于比例数量（最后一组兜底）
        {
            "name": "fewer_items_than_ratios",
            "input_list": [1, 2],  # 2个元素，3个比例
            "ratios": [0.4, 0.3, 0.3],
            "shuffle": False,
            "expected_lengths": [1, 1, 0],  # 按四舍五入分配后，最后一组无剩余
            "expected_ordered": True
        },
        # 测试用例5：处理空列表
        {
            "name": "empty_input_list",
            "input_list": [],
            "ratios": [0.5, 0.5],
            "shuffle": True,
            "expected_lengths": [0, 0],
            "expected_ordered": True  # 空列表无所谓顺序
        },
        # 测试用例6：比例总和略超1.0（允许误差范围内）
        {
            "name": "ratio_sum_slightly_over",
            "input_list": list(range(100)),  # 100个元素
            "ratios": [0.6, 0.41],  # 总和1.01，在允许误差内
            "shuffle": False,
            "expected_lengths": [60, 40],  # 最后一组会自动调整为剩余数量
            "expected_ordered": True
        },
        # 测试用例7：处理非列表的可迭代对象（如元组）
        {
            "name": "non_list_iterable_input",
            "input_list": ("a", "b", "c", "d", "e"),  # 元组输入
            "ratios": [0.4, 0.6],
            "shuffle": False,
            "expected_lengths": [2, 3],
            "expected_ordered": True
        },
        # 测试用例8：元素为复杂结构（如文件组列表）
        {
            "name": "complex_elements",
            "input_list": [
                ["img1.jpg", "lab1.txt"],
                ["img2.jpg", "lab2.txt"],
                ["img3.jpg", "lab3.txt"],
                ["img4.jpg", "lab4.txt"]
            ],
            "ratios": [0.5, 0.5],
            "shuffle": False,
            "expected_lengths": [2, 2],
            "expected_ordered": True
        }
    ],
    ids=[case["name"] for case in [
        {"name": "basic_2_8_split_no_shuffle"},
        {"name": "basic_3_3_4_split_with_shuffle"},
        {"name": "single_ratio_no_split"},
        {"name": "fewer_items_than_ratios"},
        {"name": "empty_input_list"},
        {"name": "ratio_sum_slightly_over"},
        {"name": "non_list_iterable_input"},
        {"name": "complex_elements"}
    ]]
)
def test_split_list_by_ratio(test_case_split_list_by_ratio):
    """测试split_list_by_ratio函数的各种拆分场景"""
    # 1. 准备测试参数
    input_list = test_case_split_list_by_ratio["input_list"]
    ratios = test_case_split_list_by_ratio["ratios"]
    shuffle = test_case_split_list_by_ratio["shuffle"]
    expected_lengths = test_case_split_list_by_ratio["expected_lengths"]
    expected_ordered = test_case_split_list_by_ratio["expected_ordered"]
    
    # 2. 执行被测试函数
    result = split_list_by_ratio(
        items=input_list,
        ratios=ratios,
        shuffle=shuffle
    )
    
    # 3. 验证基本结构：拆分后的子列表数量应与比例数量一致
    assert len(result) == len(ratios), (
        f"测试用例 {test_case_split_list_by_ratio['name']} 失败\n"
        f"预期子列表数量: {len(ratios)}\n"
        f"实际子列表数量: {len(result)}"
    )
    
    # 4. 验证各子列表长度是否符合预期
    actual_lengths = [len(sublist) for sublist in result]
    assert actual_lengths == expected_lengths, (
        f"测试用例 {test_case_split_list_by_ratio['name']} 失败\n"
        f"预期各子列表长度: {expected_lengths}\n"
        f"实际各子列表长度: {actual_lengths}"
    )
    
    # 5. 验证总元素数量不变（空列表除外）
    if len(input_list) > 0:
        total_actual = sum(actual_lengths)
        assert total_actual == len(input_list), (
            f"测试用例 {test_case_split_list_by_ratio['name']} 失败\n"
            f"元素总量不匹配，预期: {len(input_list)}, 实际: {total_actual}"
        )
    
    # 6. 验证元素内容是否完整（无新增、无丢失）
    #  flatten结果列表并排序
    flattened_result = [item for sublist in result for item in sublist]
    sorted_result = sorted(flattened_result)
    sorted_input = sorted(input_list)
    assert sorted_result == sorted_input, (
        f"测试用例 {test_case_split_list_by_ratio['name']} 失败\n"
        f"元素内容不匹配，预期: {sorted_input}\n"
        f"实际: {sorted_result}"
    )
    
    # 7. 验证顺序是否符合预期（未打乱时应保持原顺序）
    if not shuffle and expected_ordered and len(input_list) > 0:
        # 拼接所有子列表，检查是否与原列表一致
        concatenated = []
        for sublist in result:
            concatenated.extend(sublist)
        
        # 修复：将输入转换为列表后再比较（统一类型）
        assert concatenated == list(input_list), (
            f"测试用例 {test_case_split_list_by_ratio['name']} 失败\n"
            f"未打乱模式下顺序不一致，预期: {input_list}\n"
            f"实际: {concatenated}"
        )


@pytest.mark.parametrize(
    "test_case_invalid_inputs",
    [
        # 测试用例1：比例包含非正数
        {
            "name": "invalid_negative_ratio",
            "input_list": [1, 2, 3],
            "ratios": [0.5, -0.5],
            "shuffle": True,
            "expected_exception": ValueError,
            "exception_msg_contains": "拆分比例必须为正数"
        },
        # 测试用例2：比例包含零
        {
            "name": "invalid_zero_ratio",
            "input_list": [1, 2, 3],
            "ratios": [0.0, 1.0],
            "shuffle": False,
            "expected_exception": ValueError,
            "exception_msg_contains": "拆分比例必须为正数"
        },
        # 测试用例3：非可迭代对象作为输入
        {
            "name": "non_iterable_input",
            "input_list": 123,  # 数字不可迭代
            "ratios": [0.5, 0.5],
            "shuffle": True,
            "expected_exception": TypeError,
            "exception_msg_contains": "必须是可迭代对象"
        },
        # 测试用例4：比例不是列表
        {
            "name": "ratios_not_list",
            "input_list": [1, 2, 3],
            "ratios": "0.5,0.5",  # 字符串不是列表
            "shuffle": False,
            "expected_exception": TypeError,
            "exception_msg_contains": "ratios 必须是浮点数/整数组成的列表"
        }
    ],
    ids=[case["name"] for case in [
        {"name": "invalid_negative_ratio"},
        {"name": "invalid_zero_ratio"},
        {"name": "non_iterable_input"},
        {"name": "ratios_not_list"}
    ]]
)
def test_split_list_by_ratio_invalid_inputs(test_case_invalid_inputs):
    """测试split_list_by_ratio函数对无效输入的处理"""
    # 1. 准备测试参数
    input_list = test_case_invalid_inputs["input_list"]
    ratios = test_case_invalid_inputs["ratios"]
    shuffle = test_case_invalid_inputs["shuffle"]
    expected_exception = test_case_invalid_inputs["expected_exception"]
    exception_msg_contains = test_case_invalid_inputs["exception_msg_contains"]
    
    # 2. 执行被测试函数并验证异常
    with pytest.raises(expected_exception) as exc_info:
        split_list_by_ratio(
            items=input_list,
            ratios=ratios,
            shuffle=shuffle
        )
    
    # 3. 验证异常信息是否包含预期内容
    assert exception_msg_contains in str(exc_info.value), (
        f"测试用例 {test_case_invalid_inputs['name']} 失败\n"
        f"预期异常信息包含: {exception_msg_contains}\n"
        f"实际异常信息: {str(exc_info.value)}"
    )