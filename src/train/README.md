# lovely-deep-learning

统一的深度学习训练仓库：用一份 YAML 覆盖训练、断点续训、验证、测试、预测、导出和剪枝。核心逻辑在 `src/`，命令入口在 `scripts/`，实验配置在 `configs/experiments/`。

基于 [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)。一个实验配置同时驱动 `fit` / `validate` / `test` / `predict` / `export` / `prune`。

## 目录

- [环境安装](#环境安装)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [统一 CLI](#统一-cli)
- [实验配置](#实验配置)
- [数据集](#数据集)
- [导出与剪枝](#导出与剪枝)
- [板端转换](#板端转换)
- [训练日志](#训练日志)
- [测试](#测试)

## 环境安装

Python `>=3.10,<3.14`，依赖由 Poetry 管理。

```bash
cd src/train
poetry install
poetry shell
```

主要依赖：PyTorch 2.8、Lightning 2.5、Ultralytics、ONNX、torch-pruning。PyTorch 源已指向 CUDA 12.6 官方 wheel。

## 仓库结构

```text
src/train/
├── src/lovely_deep_learning/   # 源码
│   ├── dataset/                # CSV → Dataset
│   ├── data_module/            # Lightning DataModule
│   ├── nn/                     # 网络积木（Conv、C2f、MobileNet 等）
│   ├── model/                  # DAGNet 与权重加载
│   ├── module/                 # LightningModule（分类 / 检测）
│   ├── loss/ metric/ postprocess/
│   ├── pruning/ export/        # 剪枝与 ONNX/PT 导出
│   └── cli/                    # Lightning CLI + export/prune 子命令
├── scripts/                    # 训练入口与数据表脚本
├── configs/
│   ├── experiments/            # 一份 YAML = 一次完整实验
│   ├── models/                 # 可复用的 DAGNet 结构
│   ├── callbacks/ optim/ pruner/ transforms/
├── tests/
├── datasets/                   # 数据（不进 Git）
├── logs/                       # TensorBoard 与 checkpoint
└── exports/                    # 板端转换脚本与本地 SDK
```

约定：`module` 组合 dataset / model / loss / metric；`scripts` 只负责调用，不写训练逻辑。

## 快速开始

首次 `fit` 时，对应 DataModule 会按需下载数据并生成 CSV。

```bash
# MNIST（最小通路）
python scripts/train.py fit --config configs/experiments/image_classifiter_MNIST.yaml

# ImageNette 分类（约 99% acc）
python scripts/train.py fit --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml

# COCO8 检测（小样，适合打通流程）
python scripts/train.py fit --config configs/experiments/object_detect_COCO8.yaml
```

冒烟测试（只跑极少 step，确认数据、模型、训练循环能走通）：

```bash
python scripts/train.py fit \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --trainer.fast_dev_run true
```

## 统一 CLI

所有实验配置都走同一个入口：`python scripts/train.py <子命令> --config <yaml>`。

| 子命令 | 作用 |
|--------|------|
| `fit` | 训练；可加 `--ckpt_path` 断点续训 |
| `validate` | 验证 |
| `test` | 测试 |
| `predict` | 预测 |
| `export` | 导出 ONNX / PT（细节在 YAML 的 `exporter`） |
| `prune` | 结构化剪枝，写出 `.pth` |

下面以 ImageNette 为例。路径里的 `version_N`、checkpoint 文件名请换成实际产物。

### 训练

```bash
python scripts/train.py fit --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml
```

### 断点续训

指定 `--trainer.logger.init_args.version` 会继续写到同一日志目录，否则会新建 version。

```bash
python scripts/train.py fit \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt \
  --trainer.logger.init_args.version 0
```

`fit` 的 `--ckpt_path` 会恢复完整训练状态（权重、优化器、epoch）。`export` / `prune` 的 `--ckpt_path` 只加载模型权重。

### 验证 / 测试 / 预测

```bash
python scripts/train.py validate \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt

python scripts/train.py test \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt

python scripts/train.py predict \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt
```

Lightning CLI 参数可覆盖 YAML，例如 `--trainer.max_epochs 5`、`--trainer.fast_dev_run true`。

## 实验配置

`configs/experiments/` 下一份 YAML 就是一次实验，通常包含 `trainer`、`data`、`model`（DAGNet + optimizer / loss / exporter / pruner）。

| 配置 | 任务 | 数据 |
|------|------|------|
| `image_classifiter_MNIST.yaml` | 分类 | MNIST |
| `image_classifiter_IMAGE_NETTE.yaml` | 分类 | ImageNette（MobileNetV3） |
| `image_classifiter_IMAGE_NETTE_mobilenet_v2.yaml` | 分类 | ImageNette（MobileNetV2） |
| `image_classifiter_IMAGE_NET.yaml` | 分类 | ImageNet |
| `object_detect_COCO8.yaml` | 检测 | COCO8（YOLOv8n） |
| `object_detect_COCO.yaml` | 检测 | COCO 2017（YOLOv8n） |

公开数据集首次 `fit` 会由 DataModule 的 `prepare_data` 下载并写 CSV。COCO 2017 可将 `coco2017labels.zip`、`train2017.zip`、`val2017.zip` 预先放到 `datasets/COCO/`，存在则跳过下载。

## 数据集

数据加载一律走 CSV。CSV 中的路径相对 **CSV 所在目录** 解析，并与 YAML 里的 `key_map` 对齐。

### 图像分类

训练 / 验证 / 测试 CSV 需要：

```text
path_img,class_name,class_id
```

预测 CSV 只需：

```text
path_img
```

类别映射写在 YAML 的 `data.init_args.map_class_id_to_class_name`（dict 或指向含 `class_id,class_name` 的 CSV）。

自定义分类数据可分三步生成表：

```bash
python scripts/create_csv_to_save_path.py \
  datasets/IMAGENETTE/imagenette2-320/train \
  --relative-to datasets/IMAGENETTE \
  --header path_img \
  -o train.csv

python scripts/add_column_from_path.py \
  train.csv \
  --path-col path_img \
  --pos -2 \
  --col-name class_name

python scripts/add_class_id.py train.csv datasets/IMAGENETTE/info.yaml
```

`add_class_id.py` 会就地改写 CSV。`info.yaml` 格式为 `class_name: class_id`。

### 目标检测

训练 / 验证 / 测试 CSV 需要图像路径和 YOLO txt 标签路径：

```text
path_img,path_label_detect_yolo
```

预测 CSV 只需 `path_img`。标签为 YOLO 检测格式（每行 `class_id cx cy w h`，相对宽高归一化）。

从「图片目录 + 标签目录」按文件名配对生成表：

```bash
python scripts/create_dataset_table_from_multi_folders_for_train_predict.py --help

python scripts/create_dataset_table_from_multi_folders_for_train_predict.py \
  --headers "path_img,path_label_detect_yolo" \
  --dirs "datasets/coco8/images/train" \
  --dirs "datasets/coco8/labels/train" \
  --suffix-groups ".jpg,.png" \
  --suffix-groups ".txt" \
  --output-dir "datasets/coco8" \
  --split-ratio "1.0" \
  --output-names "train.csv" \
  --relative-to "datasets/coco8" \
  --allow-missing \
  --shuffle \
  --verbose
```

`--split-ratio 0.7,0.3` 且 `--output-names train.csv,val.csv` 可一次拆成训练 / 验证集。

## 导出与剪枝

### 导出

走 `train.py export`，不是单独的 `export.py`。`--export_format` 为 `onnx` 或 `pt`；输入尺寸、opset 等仍读 YAML 里的 `exporter`。

```bash
python scripts/train.py export \
  --config configs/experiments/object_detect_COCO8.yaml \
  --ckpt_path logs/object_detect_COCO8/version_0/checkpoints/last.ckpt \
  --export_format onnx
```

不传 `--ckpt_path` 时使用 YAML 中的初始权重。有 checkpoint 时，默认写到该 ckpt 同目录（例如 `last.onnx`）。

### 剪枝

```bash
python scripts/train.py prune \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt
```

剪枝超参在 YAML 的 `pruner.init_args`。未指定 `output_path` 时，产物为 ckpt 同目录的 `pruning{率}_{stem}.pth`。可用 CLI 覆盖剪枝率：

```bash
python scripts/train.py prune \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt \
  --model.init_args.model.pruner.init_args.tp_pruner_cfg.init_args.pruning_ratio 0.5
```

剪枝后微调：把实验 YAML 里 `weight.stages` 改成加载该 `.pth`（`format: torch_pruning`），再正常 `fit`。参考配置见 `configs/pruner/torch_pruning.yaml`。

## 板端转换

ONNX / PT 导出之后，板端量化与转换按芯片厂商放在 `exports/`。原厂 SDK 需本地下载，**不纳入 Git**。

- 目录约定：[exports/README.md](exports/README.md)
- 全志 AWNN：[exports/awnn/README.md](exports/awnn/README.md)
- 星辰 SigmaStar：[exports/sigmastar/README.md](exports/sigmastar/README.md)

## 训练日志

```bash
tensorboard --logdir logs
```

checkpoint 与 `config.yaml` 在 `logs/<实验名>/version_N/`。测试 / 预测可视化默认写到 YAML 里配置的 `tmp/`。

## 测试

```bash
pytest -m "not cli"          # 单测（不跑完整 CLI）
pytest -m cli                # CLI 集成测试（需要本地数据集）
```
