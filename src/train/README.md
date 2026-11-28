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
- [分类结果写回 FiftyOne](#分类结果写回-fiftyone)
- [测试](#测试)

## 环境安装

Python `>=3.10,<3.14`，依赖由 uv 管理。

```bash
cd src/train
uv sync
uv run python scripts/train.py fit --config configs/experiments/image_classifiter_MNIST.yaml
```

`uv sync` 创建 `.venv` 并以可编辑方式安装本包。日常命令用 `uv run`，不必再进 shell。

主要依赖：PyTorch 2.8（CUDA 12.8 官方 wheel）、Lightning 2.5、Ultralytics、ONNX、torch-pruning。TensorRT 的库包来自 NVIDIA 源，清华镜像上只有占位包。CUDA 12.8 的 wheel 同时包含 Ampere `sm_86`（4060）和 Blackwell `sm_120`（5090）。驱动需支持 CUDA 12.8，`nvidia-smi` 右上角的 CUDA Version 至少为 12.8。

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
├── datasets/                   # 数据不进 Git；各目录 README.md 说明下载与布局
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
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt \
  --trainer.logger.init_args.version 4

python scripts/train.py test \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt

python scripts/train.py predict \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt
```

`fit` 里每个 epoch 的验证和单独的 `validate` 走同一套 `validation_step`。它们每个 epoch 都要跑，只为早停和选 checkpoint，所以只记核心指标。属性分类里是 `val_loss` 和 `val_acc`：`val_acc` 是各属性在阈值 0.5 上的准确率再平均。`EarlyStopping` 看 `val_loss`，`ModelCheckpoint` 看 `val_acc`。

`test` 在训练结束后对选定权重跑一次，这里才算完整分析：逐属性 AP、在验证集上按 F1 选一次阈值，再在测试集上算精确率、召回率、F1，写出排序表，并保存误报和漏报图。这些数不参与早停，也不决定存哪个 checkpoint。

属性实验的 `AttributeClassifierMetric` 已经是这条轻路径。`fit` 里 `LogAttributeClassifierTrainValVisualizationCallback` 每个 epoch 只把训练和验证第一批的前 4 张脸写入 TensorBoard，不落盘，也不算逐属性指标。验证集每个 epoch 打乱，样本集合不变，核心指标不随顺序变化。`SaveAttributeClassifierTestPredictVisualizationCallback` 只挂在 `test` 和 `predict`。

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

各数据集的下载、目录和 CSV 位置见 [数据集](#数据集)。

## 数据集

MNIST 由 torchvision 直接读。其余数据加载走 CSV，路径相对 **CSV 所在目录** 解析，并与 YAML 里的 `key_map` 对齐。下载、解压布局和 `prepare_data` 写表的位置写在 `datasets/` 各目录的 `README.md`，总索引是 [datasets/README.md](datasets/README.md)。布局以对应类的 docstring 为准。

| 数据 | 类 | 说明 |
|------|----|------|
| MNIST | `MNISTDataModule` | [datasets/MNIST/README.md](datasets/MNIST/README.md) |
| ImageNette | `ImageNetteDataModule` | [datasets/IMAGENETTE/README.md](datasets/IMAGENETTE/README.md) |
| ImageNet | `ImageNetDataModule` | [datasets/IMAGENET/README.md](datasets/IMAGENET/README.md) |
| COCO8 | `COCO8DataModule` | [datasets/COCO8/README.md](datasets/COCO8/README.md) |
| COCO 2017 | `COCODataModule` | [datasets/COCO/README.md](datasets/COCO/README.md) |
| 框级分类 | `ImageClassifierFromDetectBoxDataModule` | [datasets/box_crop.md](datasets/box_crop.md) |

### 图像分类

训练 / 验证 / 测试 CSV 需要：

```text
path_img,class_name,class_id
```

预测 CSV 只需：

```text
path_img
```

训练哪些 id 由 `data.init_args.class_groups` 决定（`{训练 id: [原始 class_id, ...]}`）。不写则 CSV 里出现的 `class_id` 全部训练。绘图用的名称由数据集按筛选结果生成；某个 id 没有名字时，图上写这个 id。检测任务的类别名在 `map_class_id_to_class_name.csv`，由 COCO / COCO8 的 `prepare_data` 写出，YAML 把该路径传给 `map_class_id_to_class_name`。

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

`--split-ratio 0.7,0.3` 且 `--output-names train.csv,val.csv` 可一次拆成训练 / 验证集。COCO8 / COCO 2017 的检测 CSV 由对应 DataModule 生成，不必再跑这个脚本。

### 框级分类

`ImageClassifierFromDetectBoxDataModule` 不下载、也不生成 CSV。先用 `scripts/yolo_dir_to_box_crop_csv.py` 把 YOLO 目录展开成表，列和实验配置见 [datasets/box_crop.md](datasets/box_crop.md)。

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

## 分类结果写回 FiftyOne

对 FiftyOne 里已有的框打分。`--sample-tags` 选图。`--class-map` 把模型输出 id 一一对应到框上的 tag，例如 `0:face_frontal,1:face_side,2:face_head`。`--box-labels head` 已经选定框。映射值必须和框上的 tag 完全一致，例如 `face_frontal`，不要写 `head-face_frontal`。带这些样本 tag 的图上，`--box-labels` 里的框都会看。短边小于配置 `min_side_px`、框无效、或数据集筛掉的框不打分，写 `eval_skip`。正好命中一个映射 tag 才打分：判错写 `eval_err` 和 `pred_<tag>`，判对只写预测标签。一个映射 tag 都没有的框也推理，写 `eval_miss_tag` 和 `pred_<tag>`，不进错误率，预测可当预标注。命中多个映射 tag 时只写 `eval_mult_tag`，不进模型。预测 id 不在映射里时，标签是 `pred_<id>`。不改真值 tag 和 `changed`。`--ckpt` 须与 `--config` 的分类头一致。

本机若已有 mongod 在 27017，脚本会默认连 `mongodb://127.0.0.1:27017`（也可用环境变量 `FIFTYONE_DATABASE_URI` 覆盖）。

```bash
uv run scripts/val_classifier_with_fiftyone.py \
  --ckpt logs/image_classifiter_from_detect_box_head_face_mobilenet_v2_v001/version_0/checkpoints/last.ckpt \
  --config configs/experiments/image_classifiter_from_detect_box_head_face_mobilenet_v2_v001.yaml \
  --dataset BBM08S_head \
  --sample-tags test_v401 \
  --box-labels head \
  --class-map 0:face_frontal,1:face_side,2:face_head \
  --write-fo

uv run fiftyone app launch BBM08S_head --address 0.0.0.0 --port 5151
```

左侧 **Label tags** 同时勾选两个框标签是「有其中一个即可」。要同时具备，用图片网格上方的视图栏：

1. 点视图栏右侧 **+**，输入 `FilterLabels` 并回车。
2. **field** 填 `ground_truth`，回车。
3. **filter** 粘贴 JSON（页面不接受 Python 的 `F(...)`），回车：

```json
{"$setIsSubset": [["eval_err", "pred_face_frontal"], "$$this.tags"]}
```

4. **only_matches** 保持 `True`，**trajectories** 保持 `False`，各回车一次。

网格只留下同时带这两个标签的框。其它预测类把 `pred_face_frontal` 换成对应的 `pred_<tag>`。没有映射 tag 的框用 `eval_miss_tag` 筛选，命中多个映射 tag 的框用 `eval_mult_tag` 筛选，跳过的框用 `eval_skip` 筛选。左上角视图下拉里的 **Save current filters as view** 可保存；视图栏右侧清空按钮回到完整数据集。

## 测试

```bash
uv run pytest -m "not cli"   # 单测（不跑完整 CLI）
uv run pytest -m cli         # CLI 集成测试（需要本地数据集）

uv run scripts/yolo_dir_to_box_crop_csv.py --data-root /home/huangwenhua/project/dataset/eye/v301_eye
```
