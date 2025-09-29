# -*- encoding: utf-8 -*-
"""
@File    :   get_txt_for_dataset.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/1/13 23:33:13
@Author  :   xiaopangdun
@Email   :   18675381281@163.com
@Desc    :   目的是自动化生成用于模型训练的数据集，并确保数据集的正确分割和保存。
"""

import typer
import csv
from pathlib import Path
from typing import List, Optional

from lovely_deep_learning.utils.file import list_grouped_files_from_folders, split_list_by_ratio


app = typer.Typer(
    help="将多个文件夹中的文件按基础名分组，并按比例拆分为多个CSV文件",
)


def parse_comma_list(s: str) -> List[str]:
    """解析逗号分隔的字符串为列表，处理空格"""
    return [part.strip() for part in s.split(",")]


def parse_split_ratios(s: str) -> List[float]:
    """解析拆分比例为浮点数列表"""
    try:
        ratios = [float(ratio.strip()) for ratio in s.split(",")]
        # 检查比例是否为正数
        for ratio in ratios:
            if ratio <= 0:
                raise ValueError("拆分比例必须为正数")
        return ratios
    except ValueError as e:
        raise typer.BadParameter(f"拆分比例格式错误：{str(e)}")


def ensure_directory(path: Path):
    """确保目录存在，不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


# 辅助函数（保持不变）
def validate_split_param_coexistence(split_ratio: Optional[str], output_names: Optional[str]):
    if (split_ratio is None) != (output_names is None):
        typer.echo(
            typer.style("错误：--split-ratio 和 --output-names 必须同时提供或同时不提供", fg=typer.colors.RED), err=True
        )
        raise typer.Exit(code=1)


def validate_parameter_counts(dirs: List[Path], suffix_groups: List[List[str]], headers: List[str]):
    if not (len(dirs) == len(suffix_groups) == len(headers)):
        typer.echo(
            typer.style(
                f"错误：参数数量不匹配\n"
                f"- 文件夹数量：{len(dirs)}\n"
                f"- 后缀组数量：{len(suffix_groups)}\n"
                f"- 表头数量：{len(headers)}",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)


def process_output_names(names: List[str], verbose: bool) -> List[str]:
    processed_names = []
    for name in names:
        if not name.endswith(".csv"):
            name = f"{name}.csv"
            if verbose:
                typer.echo(f"文件名自动补充.csv后缀：{name}")
        processed_names.append(name)
    return processed_names


def validate_split_details(split_ratios: List[float], output_names: List[str]):
    if len(split_ratios) != len(output_names):
        typer.echo(
            typer.style(
                f"错误：拆分比例数量（{len(split_ratios)}）与输出文件名数量（{len(output_names)}）不匹配",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    ratio_sum = sum(split_ratios)
    if not (0.999 <= ratio_sum <= 1.001):
        typer.echo(typer.style(f"错误：拆分比例总和为 {ratio_sum:.3f}，需接近1.0", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)


def get_file_groups(
    dirs: List[Path], suffix_groups: List[List[str]], relative_to: Optional[Path], allow_missing: bool, verbose: bool
) -> List[List[str]]:
    if verbose:
        typer.echo("开始处理文件分组...")

    grouped_files = list_grouped_files_from_folders(
        dirs=dirs, suffix_groups=suffix_groups, relative_to=relative_to, allow_missing=allow_missing, verbose=verbose
    )

    total_groups = len(grouped_files)
    if verbose:
        typer.echo(f"成功生成 {total_groups} 个文件组")

    return grouped_files


def write_split_results(
    split_groups_list: List[List[List[str]]],
    output_names: List[str],
    output_dir: Path,
    headers: List[str],
    verbose: bool,
):
    for name, groups in zip(output_names, split_groups_list):
        file_path = output_dir / name
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(groups)
        if verbose:
            typer.echo(f"已保存：{file_path}（{len(groups)}组）")


@app.command()
def group_files(
    headers: str = typer.Option(
        "data_img,label_detect_yolo", "--headers", help="CSV表头（逗号分隔，如 '图像路径,标注路径'）"
    ),
    dirs: List[Path] = typer.Option(
        ["datasets/coco8/images/train", "datasets/coco8/labels/train"],
        "--dirs",
        help="目标文件夹路径（可重复输入，如 --dirs ./img --dirs ./lab）",
    ),
    suffix_groups: List[str] = typer.Option(
        [".jpg,.png", ".txt"], "--suffix-groups", "-suf", help="每个文件夹允许的文件后缀（可重复输入，每组用逗号分隔）"
    ),
    output_dir: Path = typer.Option(
        "/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8",
        "--output-dir",
        help="拆分后CSV文件的保存目录",
    ),
    split_ratio: Optional[str] = typer.Option(
        "0.5,0.5",
        "--split-ratio",
        help="按逗号分隔的拆分比例（小数），如 --split-ratio 0.7,0.2,0.1，单比例1.0表示不拆分",
    ),
    output_names: Optional[str] = typer.Option(
        "train.csv,val.csv",
        "--output-names",
        help="按逗号分隔的输出文件名（需含.csv），如 --output-names train.csv,val.csv,test.csv",
    ),
    relative_to: Optional[Path] = typer.Option(None, "--relative-to", help="输出路径相对于此目录"),
    allow_missing: bool = typer.Option(True, "--allow-missing/--no-allow-missing", help="是否允许分组中存在缺失文件"),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle", help="拆分前是否打乱文件组顺序"),
    verbose: bool = typer.Option(True, "--verbose/--no-verbose", "-v", help="显示详细的处理信息"),
):
    """
    将多个文件夹中的文件按基础文件名分组，支持按比例拆分为多个CSV文件（如训练/验证/测试集）。

    常用示例：

    1. 不拆分，生成单个文件：
        python get_txt_for_dataset.py --split-ratio 1.0 --output-names total.csv

    2. 7:3拆分训练/验证集：
        python get_txt_for_dataset.py --split-ratio 0.7,0.3 --output-names train.csv,val.csv

    3. 简化输出路径并关闭详细日志：
        python get_txt_for_dataset.py --relative-to ./datasets --no-verbose
    """
    try:
        # 校验拆分参数共存性
        validate_split_param_coexistence(split_ratio, output_names)

        # 解析基础参数
        headers_list = parse_comma_list(headers)
        parsed_suffix_groups = [parse_comma_list(group) for group in suffix_groups]

        # 校验基础参数数量
        validate_parameter_counts(dirs, parsed_suffix_groups, headers_list)

        # 确保输出目录存在
        ensure_directory(output_dir)

    except Exception as e:
        typer.echo(typer.style(f"参数解析错误：{str(e)}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    try:
        # 解析拆分参数
        split_ratios = parse_split_ratios(split_ratio)
        output_names_list = parse_comma_list(output_names)

        # 校验拆分参数细节
        validate_split_details(split_ratios, output_names_list)

        # 处理输出文件名
        output_names_list = process_output_names(output_names_list, verbose)

    except typer.BadParameter as e:
        typer.echo(typer.style(str(e), fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(typer.style(f"拆分参数解析错误：{str(e)}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    try:
        # 获取文件分组
        grouped_files = get_file_groups(
            dirs=dirs,
            suffix_groups=parsed_suffix_groups,
            relative_to=relative_to,
            allow_missing=allow_missing,
            verbose=verbose,
        )

    except Exception as e:
        typer.echo(typer.style(f"处理文件分组时出错：{str(e)}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    try:
        # 输出拆分统计（无论是否拆分，统一显示）
        if verbose:
            typer.echo("\n拆分统计：")
            typer.echo(f"总文件组数：{len(grouped_files)}")
            typer.echo(f"拆分比例：{split_ratios}")

        # 执行拆分（单比例1.0时也会正常处理，结果仍是完整列表）
        split_groups_list = split_list_by_ratio(grouped_files, split_ratios, shuffle)

        # 打印分配数量
        if verbose:
            typer.echo(f"分配数量：{[len(g) for g in split_groups_list]}")

        # 写入CSV文件
        write_split_results(
            split_groups_list=split_groups_list,
            output_names=output_names_list,
            output_dir=output_dir,
            headers=headers_list,
            verbose=verbose,
        )

        typer.echo(typer.style("\n处理完成！", fg=typer.colors.GREEN))

    except PermissionError:
        typer.echo(typer.style(f"权限不足，无法写入目录：{output_dir}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(typer.style(f"写入文件时出错：{str(e)}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
