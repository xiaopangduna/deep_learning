# ImageNette

布局以 `lovely_deep_learning.data_module.image_nette.ImageNetteDataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.image_nette.ImageNetteDataModule`
- 实验：`configs/experiments/image_classifiter_IMAGE_NETTE.yaml`、`configs/experiments/image_classifiter_IMAGE_NETTE_mobilenet_v2.yaml`

## 怎么得到数据

`prepare_data` 在 `dataset_dir` 不存在时下载并解压。压缩包留在父目录，不删除。

- URL：`https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz`
- 下载到：`datasets/IMAGENETTE/imagenette2-320.tgz`
- 解压到：`datasets/IMAGENETTE/`（包内顶层目录就是 `imagenette2-320/`）

`dataset_dir` 已存在则跳过下载。`train.csv`、`val.csv`、`predict.csv` 三个都在时跳过生成。

## 目录

`dataset_dir` 默认为 `datasets/IMAGENETTE/imagenette2-320`。类别 id 按 `train/` 下文件夹名字典序分配。CSV 写在父目录，`path_img` 相对 CSV 所在目录。

```text
datasets/IMAGENETTE/
├── imagenette2-320.tgz
├── imagenette2-320/
│   ├── train/<class_name>/*.JPEG
│   └── val/<class_name>/*.JPEG
├── train.csv
├── val.csv
└── predict.csv
```

没有单独的 `test.csv`。实验配置里的 `test_csv_paths` 指向 `val.csv`。`predict.csv` 只有 `path_img`，内容来自验证集图像。

## CSV

`train.csv` / `val.csv`：

```text
path_img,class_name,class_id
imagenette2-320/train/n01440764/xxx.JPEG,n01440764,0
```

`predict.csv`：

```text
path_img
imagenette2-320/val/n01440764/xxx.JPEG
```

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.image_nette.ImageNetteDataModule
  init_args:
    dataset_dir: datasets/IMAGENETTE/imagenette2-320
    train_csv_paths: [datasets/IMAGENETTE/train.csv]
    val_csv_paths: [datasets/IMAGENETTE/val.csv]
    test_csv_paths: [datasets/IMAGENETTE/val.csv]
    predict_csv_paths: [datasets/IMAGENETTE/predict.csv]
```
