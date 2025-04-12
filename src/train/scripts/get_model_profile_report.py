# -*- encoding: utf-8 -*-
'''
@File    :   get_model_profile_report.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/03/23 15:41:39
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
'''
import os
import sys

path_parent_dir = os.path.dirname(sys.path[0])
sys.path.insert(0, path_parent_dir)

import argparse
from pathlib import Path
from typing import List
import random
import logging
from collections import defaultdict
import argparse
import importlib
import torch
import sys
import os
from src.utils.profiler_pytorch import compare_models
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")




import argparse
import torch
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="模型性能对比工具参数化调用脚本")
    parser.add_argument("--model_paths", nargs="+", default=["logs/DirectionalCornerDetectionModel/version_1/checkpoints/last.pth"],
                        help="模型 .pth 文件路径列表，例如 path/to/model1.pth path/to/model2.pth")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="模型名称列表，与模型路径一一对应，例如 ModelA ModelB")
    parser.add_argument("--input_shape", type=str, default="1,3,512,512",
                        help="输入张量形状，格式为逗号分隔的整数，例如 '32,10'")
    parser.add_argument("--use_gpu", default=True,
                        help="启用 GPU 加速")
    parser.add_argument("--warmup_runs", type=int, default=3,
                        help="热身运行次数，默认 3")
    parser.add_argument("--actual_runs", type=int, default=10,
                        help="实际测量运行次数，默认 10")
    parser.add_argument("--save_folder", default="profiling_results",
                        help="报告保存路径，默认 profiling_results")
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析输入形状
    try:
        input_shape = list(map(int, args.input_shape.split(',')))
    except ValueError:
        sys.exit("错误：输入形状格式不正确，请使用逗号分隔的整数，例如 '32,10'")

    # 生成输入张量
    input_tensor = torch.randn(*input_shape)
    if args.use_gpu and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # 确保保存路径存在
    os.makedirs(args.save_folder, exist_ok=True)

    # 加载模型
    models = {}
    for model_name, model_path in zip(args.model_names, args.model_paths):
        try:
            model = torch.load(model_path)
            models[model_name] = model
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")

    # 执行性能对比
    compare_models(
        models=models,
        input_tensor=input_tensor,
        use_gpu=args.use_gpu,
        warmup_runs=args.warmup_runs,
        actual_runs=args.actual_runs,
        save_folder=args.save_folder
    )


if __name__ == "__main__":
    main()