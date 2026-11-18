# ImageNet

布局以 `lovely_deep_learning.data_module.image_net.ImageNetDataModule` 为准。CSV 由 `generate_imagenet_csvs` 生成，`prepare_data` 和本目录的 `imagenet_to_csv.py` 都调用它。

## 类与配置

- 类：`lovely_deep_learning.data_module.image_net.ImageNetDataModule`
- 实验：`configs/experiments/image_classifiter_IMAGE_NET.yaml`、`configs/experiments/image_classifiter_IMAGE_NET_mobilenet_v2.yaml`

## 下载

官方 ILSVRC CLS-LOC 可在 [image-net.org](https://image-net.org/) 注册后下载。同一份数据在 Kaggle：

[ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)

```bash
kaggle competitions download -c imagenet-object-localization-challenge
```

这条命令得到 `imagenet-object-localization-challenge.zip`。把它放到 `datasets/IMAGENET/imagenet/`（见下一节；该目录可以是软链接）。训练图片约 138GB，不要拷进 Git。

## 解压

在 `datasets/IMAGENET/imagenet/` 里解压。zip 和解压结果都留在这一层。zip 里直接是 `ILSVRC/`，没有再套一层 tar.gz。其中约 143 万个成员是 JPEG。不带筛选地 `unzip` 会把图片一并写出来。

只解开标注、ImageSets 和那几个 `LOC_*.csv` / `LOC_*.txt`：

```bash
cd datasets/IMAGENET/imagenet
unzip -q imagenet-object-localization-challenge.zip -d ./ \
  -x "ILSVRC/Data/*"
```

图片按需再抽。验证，以及实验里的测试，读的是 val，约 6GB：

```bash
unzip -q imagenet-object-localization-challenge.zip -d ./ \
  "ILSVRC/Data/CLS-LOC/val/*"
```

训练时再抽 train，约 138GB：

```bash
unzip -q imagenet-object-localization-challenge.zip -d ./ \
  "ILSVRC/Data/CLS-LOC/train/*"
```

test 约 13GB。当前实验不读 test 图，需要时再抽：

```bash
unzip -q imagenet-object-localization-challenge.zip -d ./ \
  "ILSVRC/Data/CLS-LOC/test/*"
```

已经用无筛选的 `unzip` 写出的 `ILSVRC/Data` 可以删掉，再按上面的命令重抽。

## 软链接

`datasets/IMAGENET/` 留在项目里，只放这份 README 和 `imagenet_to_csv.py`。压缩包、`LOC_*`、`ILSVRC/` 和生成的 CSV 都在 `imagenet/`。数据在别的磁盘时，把整个数据目录链成 `imagenet`，不要只链 `ILSVRC`，也不要把 `datasets/IMAGENET` 本身换成软链接。

```bash
ln -s /data/imagenet datasets/IMAGENET/imagenet
```

`/data/imagenet` 里面就是 zip 和 `ILSVRC/`。CSV 里的 `path_img` 相对 `imagenet/`，写成 `ILSVRC/Data/CLS-LOC/...`，不会把软链接展开成绝对路径。

## 生成 CSV

ImageSets 和 val XML 解开并完成软链接后即可生成。脚本不打开 JPEG，图片可以还在压缩包里。`class_id` 是 `train_cls.txt` 里 synset 的字典序；官方列表有全部 1000 类时，与 train 文件夹名字典序相同。

`train.csv` 来自 `train_cls.txt`。`val.csv` 的类别来自对应 XML 的第一个 object name；XML 缺失、解析失败或 synset 不在 `train_cls.txt` 里的行会跳过。`test.csv` 与 `predict.csv` 都只有 `path_img`，来自 `test.txt`。测试集没有标签，实验 YAML 的 `test_csv_paths` 指向 `val.csv`。

```bash
python datasets/IMAGENET/imagenet_to_csv.py
python datasets/IMAGENET/imagenet_to_csv.py --overwrite
```

四个 CSV 都已存在时跳过写入，但每次仍检查 ImageSets 和 val 标注。首次 `fit` 的 `prepare_data` 做同样的事，不必为了训练再跑一遍脚本。`--overwrite` 用于这些小文件变了之后重写。没有对应 JPEG 的行也会写进表，读图时才会失败。只有 val 图片时可以跑 `validate` / `test`。

缺下面任一路径会在生成时抛 `FileNotFoundError`。`imagenet` 若是断掉的软链接，会报 `dataset_dir` 不存在。

- `ILSVRC/Annotations/CLS-LOC/val`
- `ILSVRC/ImageSets/CLS-LOC/train_cls.txt`
- `ILSVRC/ImageSets/CLS-LOC/val.txt`
- `ILSVRC/ImageSets/CLS-LOC/test.txt`

## 目录

`path_img` 相对 `datasets/IMAGENET/imagenet`（也就是 CSV 所在目录）。`Data/CLS-LOC/` 下的图片目录在抽过对应压缩包成员之后才出现。

```text
datasets/IMAGENET/
├── README.md
├── imagenet_to_csv.py
└── imagenet -> /data/imagenet
    ├── imagenet-object-localization-challenge.zip
    ├── LOC_synset_mapping.txt
    ├── LOC_train_solution.csv
    ├── LOC_val_solution.csv
    ├── LOC_sample_submission.csv
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
    dataset_dir: datasets/IMAGENET/imagenet
    train_csv_paths: [datasets/IMAGENET/imagenet/train.csv]
    val_csv_paths: [datasets/IMAGENET/imagenet/val.csv]
    test_csv_paths: [datasets/IMAGENET/imagenet/val.csv]
    predict_csv_paths: [datasets/IMAGENET/imagenet/predict.csv]
```
