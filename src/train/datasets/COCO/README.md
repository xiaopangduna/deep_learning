# COCO 2017

布局以 `lovely_deep_learning.data_module.coco.COCODataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.coco.COCODataModule`
- 实验：`configs/experiments/object_detect_COCO.yaml`

## 怎么得到数据

压缩包放在 `dataset_dir` 的父目录，默认 `datasets/COCO/`。文件名与下表一致时直接使用，不重新下载。解压后保留 zip。

| 文件 | 来源 | 约大小 | 自动下载 |
|------|------|--------|----------|
| `coco2017labels.zip` | `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip` | 较小 | 是 |
| `val2017.zip` | `http://images.cocodataset.org/zips/val2017.zip` | 1GB | 是 |
| `train2017.zip` | `http://images.cocodataset.org/zips/train2017.zip` | 19GB | 是 |
| `test2017.zip` | `http://images.cocodataset.org/zips/test2017.zip` | 6GB | 否 |

`test2017` 只有两种情况会解压：父目录里已经有 `test2017.zip`，或 YAML 里 `download_test2017: true`。检测训练用的是 train / val，可以不准备 test。

图像目录里已经有图、或标签目录里已经有 txt 时，跳过对应解压。`train.csv`、`val.csv`、`predict.csv` 都在时跳过生成；仅缺 `map_class_id_to_class_name.csv` 时补写映射。

## 目录

`dataset_dir` 默认为 `datasets/COCO/coco`。标签 zip 解压后会被整理成 `labels/train2017` 与 `labels/val2017`。CSV 写在父目录，路径前缀是目录名 `coco`。

```text
datasets/COCO/
├── coco2017labels.zip
├── train2017.zip
├── val2017.zip
├── test2017.zip                  # 可选
├── coco/
│   ├── images/train2017/*.jpg
│   ├── images/val2017/*.jpg
│   ├── images/test2017/*.jpg     # 仅解压了 test2017 时有
│   ├── labels/train2017/*.txt
│   └── labels/val2017/*.txt
├── train.csv
├── val.csv
├── predict.csv
└── map_class_id_to_class_name.csv
```

`predict.csv` 只有验证集的 `path_img`。实验里 `test_csv_paths` 指向 `val.csv`。没有对应 txt 的图像不进 train / val CSV。

## CSV

`train.csv` / `val.csv`：

```text
path_img,path_label_detect_yolo
coco/images/train2017/000000000009.jpg,coco/labels/train2017/000000000009.txt
```

标签是 YOLO 检测格式，每行 `class_id cx cy w h`。`map_class_id_to_class_name.csv` 为 COCO 80 类，`0,person` 起。

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.coco.COCODataModule
  init_args:
    dataset_dir: datasets/COCO/coco
    train_csv_paths: [datasets/COCO/train.csv]
    val_csv_paths: [datasets/COCO/val.csv]
    test_csv_paths: [datasets/COCO/val.csv]
    predict_csv_paths: [datasets/COCO/predict.csv]
    map_class_id_to_class_name: datasets/COCO/map_class_id_to_class_name.csv
    # download_test2017: true
```
