# ImageNet

布局以 `lovely_deep_learning.data_module.image_net.ImageNetDataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.image_net.ImageNetDataModule`
- 实验：`configs/experiments/image_classifiter_IMAGE_NET.yaml`、`configs/experiments/image_classifiter_IMAGE_NET_mobilenet_v2.yaml`

## 怎么得到数据

`prepare_data` 不下载。需要自行准备 ILSVRC CLS-LOC（[image-net.org](https://image-net.org/) 注册后获取），并按下面的树放到 `dataset_dir`。默认 `datasets/IMAGENET`，其下要有 `ILSVRC/`。

四个 CSV 都已存在时跳过生成。缺任何一个目录或 ImageSets 文件会在生成时抛 `FileNotFoundError`：

- `ILSVRC/Data/CLS-LOC/train`
- `ILSVRC/Data/CLS-LOC/val`
- `ILSVRC/Data/CLS-LOC/test`
- `ILSVRC/Annotations/CLS-LOC/val`
- `ILSVRC/ImageSets/CLS-LOC/train_cls.txt`
- `ILSVRC/ImageSets/CLS-LOC/val.txt`
- `ILSVRC/ImageSets/CLS-LOC/test.txt`

## 目录

类别 id 按 `train/` 下 synset 文件夹名字典序分配，与网上常见的 0–999 固定顺序不一定相同。`path_img` 相对 `dataset_dir`（也就是 CSV 所在目录）。

```text
datasets/IMAGENET/
├── ILSVRC/
│   ├── Data/CLS-LOC/
│   │   ├── train/<synset>/*.JPEG
│   │   ├── val/*.JPEG
│   │   └── test/*.JPEG
│   ├── Annotations/CLS-LOC/val/*.xml
│   └── ImageSets/CLS-LOC/
│       ├── train_cls.txt
│       ├── val.txt
│       └── test.txt
├── train.csv
├── val.csv
├── test.csv
└── predict.csv
```

`train.csv` 来自 `train_cls.txt`。`val.csv` 的类别来自对应 XML 的第一个 object name；XML 缺失、解析失败或 synset 不在训练集里的行会跳过。`test.csv` 与 `predict.csv` 都只有 `path_img`，来自 `test.txt`。测试集没有标签，实验 YAML 的 `test_csv_paths` 指向 `val.csv`。

## CSV

`train.csv` / `val.csv`：

```text
path_img,class_name,class_id
ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_18.JPEG,n01440764,0
```

`test.csv` / `predict.csv`：

```text
path_img
ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000001.JPEG
```

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.image_net.ImageNetDataModule
  init_args:
    dataset_dir: datasets/IMAGENET
    train_csv_paths: [datasets/IMAGENET/train.csv]
    val_csv_paths: [datasets/IMAGENET/val.csv]
    test_csv_paths: [datasets/IMAGENET/val.csv]
    predict_csv_paths: [datasets/IMAGENET/predict.csv]
```
