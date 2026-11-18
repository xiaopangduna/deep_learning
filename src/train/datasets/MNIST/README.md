# MNIST

布局以 `lovely_deep_learning.data_module.mnist.MNISTDataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.mnist.MNISTDataModule`
- 实验：`configs/experiments/image_classifiter_MNIST.yaml`

这个 DataModule 不走 CSV，直接用 torchvision。

## 怎么得到数据

`prepare_data` 调用 `torchvision.datasets.MNIST(..., download=True)`，分别拉取官方 train 与 test。默认根目录是 `datasets/MNIST`。已有文件则 torchvision 跳过下载。

## 目录

torchvision 会在 `data_dir` 下再建一层 `MNIST/`：

```text
datasets/MNIST/
└── MNIST/
    ├── raw/
    └── processed/
```

## 划分

官方 train（60000）在 `setup` 里划成 55000 训练和 5000 验证。测试集用官方 test（10000）。这里没有 `train.csv`。

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.mnist.MNISTDataModule
  init_args:
    data_dir: datasets/MNIST
```
