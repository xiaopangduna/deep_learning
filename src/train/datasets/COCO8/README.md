# COCO8

布局以 `lovely_deep_learning.data_module.coco.COCO8DataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.coco.COCO8DataModule`
- 实验：`configs/experiments/object_detect_COCO8.yaml`

这是 Ultralytics 的 8 张图小样，用来打通检测流程。类别映射仍是 COCO 80 类。

## 怎么得到数据

`dataset_dir/images/train` 或 `images/val` 不存在时，`prepare_data` 下载 zip、解压到父目录，然后删除 zip。

- URL：`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip`
- 解压位置：`datasets/COCO8/`（包内顶层目录是 `coco8/`）

`train.csv`、`val.csv`、`predict.csv` 都在时跳过生成。三个主 CSV 已在、但缺 `map_class_id_to_class_name.csv` 时会补写映射。

## 目录

`dataset_dir` 默认为 `datasets/COCO8/coco8`。CSV 写在父目录。`path_img` 和 `path_label_detect_yolo` 相对 CSV 所在目录，前缀是 `dataset_dir` 的目录名。

```text
datasets/COCO8/
├── coco8/
│   ├── images/train/*.jpg
│   ├── images/val/*.jpg
│   ├── labels/train/*.txt
│   └── labels/val/*.txt
├── train.csv
├── val.csv
├── predict.csv
└── map_class_id_to_class_name.csv
```

没有 `test` 划分。实验里 `test_csv_paths` 指向 `val.csv`。`predict.csv` 只有验证集的 `path_img`。没有对应 txt 的图像不会写入 CSV。

## CSV

`train.csv` / `val.csv`：

```text
path_img,path_label_detect_yolo
coco8/images/train/000000000009.jpg,coco8/labels/train/000000000009.txt
```

标签是 YOLO 检测格式，每行 `class_id cx cy w h`（相对宽高归一化）。

`map_class_id_to_class_name.csv`：

```text
class_id,class_name
0,person
```

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.coco.COCO8DataModule
  init_args:
    dataset_dir: datasets/COCO8/coco8
    train_csv_paths: [datasets/COCO8/train.csv]
    val_csv_paths: [datasets/COCO8/val.csv]
    test_csv_paths: [datasets/COCO8/val.csv]
    predict_csv_paths: [datasets/COCO8/predict.csv]
    map_class_id_to_class_name: datasets/COCO8/map_class_id_to_class_name.csv
```
