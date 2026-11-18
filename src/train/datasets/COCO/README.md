# COCO 2017

布局以 `lovely_deep_learning.data_module.coco.COCODataModule` 为准。CSV 由 `generate_coco_csvs` 生成，`prepare_data` 和本目录的 `coco_to_csv.py` 都调用它。

## 类与配置

- 类：`lovely_deep_learning.data_module.coco.COCODataModule`
- 实验：`configs/experiments/object_detect_COCO.yaml`

## 自动准备

`fit`、`validate`、`test`、`predict` 都会调用 `prepare_data`。它在 `dataset_dir`（默认 `datasets/COCO/coco`）里一次准备标签、train2017、val2017 和 test2017：

1. 缺少 `coco2017labels.zip`、`train2017.zip`、`val2017.zip`、`test2017.zip` 时下载。已有且是完整 zip 则直接用。下到一半的文件从已有字节续传。
2. 标签包去掉顶层 `coco/` 后全部解压（`labels/`、`LICENSE`、`README.txt`、`train2017.txt`、`val2017.txt` 等）。`labels/train2017` 与 `labels/val2017` 的 txt 数量已和压缩包一致时跳过。
3. 图像包解压到 `images/`，得到 `images/train2017`、`images/val2017`、`images/test2017`。本地文件数不少于压缩包内文件数时跳过；不足则补齐，已有文件不覆盖。
4. 调用 `generate_coco_csvs`。

解压后保留 zip。

## 手动下载

官方图像在 [cocodataset.org](https://cocodataset.org/#download)。YOLO 标签用 Ultralytics 的 `coco2017labels.zip`。

| 文件 | 来源 | 约大小 |
|------|------|--------|
| `coco2017labels.zip` | `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip` | 较小 |
| `val2017.zip` | `http://images.cocodataset.org/zips/val2017.zip` | 1GB |
| `train2017.zip` | `http://images.cocodataset.org/zips/train2017.zip` | 19GB |
| `test2017.zip` | `http://images.cocodataset.org/zips/test2017.zip` | 6GB |

压缩包放到 `datasets/COCO/coco/`（见下一节；该目录可以是软链接）。文件名与上表一致时，`prepare_data` 不再下载。

## 手动解压

在 `datasets/COCO/coco/` 里解压。zip 和解压结果都留在这一层。图像 zip 的顶层目录是 `train2017/`、`val2017/`、`test2017/`，解压到 `images/` 下就是目标布局。

```bash
cd datasets/COCO/coco
unzip -n -q train2017.zip -d images
unzip -n -q val2017.zip -d images
unzip -n -q test2017.zip -d images
```

标签 zip 的成员在 `coco/` 下。去掉这一层后全部放进数据目录，包含 `labels/`、说明文件和图像列表：

```bash
unzip -q coco2017labels.zip -d /tmp/coco2017labels
cp -a /tmp/coco2017labels/coco/. .
rm -rf /tmp/coco2017labels
```

解压后是 `labels/train2017/*.txt` 与 `labels/val2017/*.txt`。

## 软链接

`datasets/COCO/` 留在项目里，只放这份 README 和 `coco_to_csv.py`。压缩包、`images/`、`labels/` 和生成的 CSV 都在 `coco/`。数据在别的磁盘时，把整个数据目录链成 `coco`，不要只链 `images` 或 `labels`，也不要把 `datasets/COCO` 本身换成软链接。

```bash
ln -s /data/coco datasets/COCO/coco
```

`/data/coco` 里面就是 zip、`images/` 和 `labels/`。CSV 里的路径相对 `coco/`，写成 `images/...` 和 `labels/...`，不会把软链接展开成绝对路径。

## 生成 CSV

`train2017.txt` 与 `val2017.txt` 就位后即可生成。这两份列表来自标签包，分别是 118287 和 5000 张图，和 Ultralytics `coco.yaml` 用的图像列表相同。脚本不打开 JPEG，图片可以还在压缩包里。

列表里的每一行都写入。有 txt 的图带上 `labels/{split}/{stem}.txt`。没有 txt 的图也写入，标签路径仍按这个规则书写；读的时候文件不存在，标签是空的。Ultralytics `verify_image_label` 同样把缺失的标签文件和空文件都当成没有目标，并算进 background。标签包里的 txt 都非空，背景就是这 1021 张训练图和 48 张验证图。

`predict.csv` 只有验证集的 `path_img`，包含没有标签的验证图。`map_class_id_to_class_name.csv` 是 COCO 80 类，`0,person` 起。test2017 没有检测标签，实验 YAML 的 `test_csv_paths` 指向 `val.csv`。

```bash
python datasets/COCO/coco_to_csv.py
python datasets/COCO/coco_to_csv.py --overwrite
```

四个文件都已存在时跳过写入，但每次仍检查 `train2017.txt` 和 `val2017.txt`。`prepare_data` 在解压之后做同样的事，不必为了训练再跑一遍脚本。`--overwrite` 用于标签变了之后重写。没有对应 JPEG 的行也会写进表，读图时才会失败。

缺 `train2017.txt` 或 `val2017.txt` 会在生成时抛 `FileNotFoundError`。`coco` 若是断掉的软链接，会报 `dataset_dir` 不存在。

## 目录

`path_img` 和 `path_label_detect_yolo` 相对 `datasets/COCO/coco`（也就是 CSV 所在目录）。`images/` 下的图片目录在抽过对应压缩包之后才出现。

```text
datasets/COCO/
├── README.md
├── coco_to_csv.py
└── coco -> /data/coco
    ├── coco2017labels.zip
    ├── train2017.zip
    ├── val2017.zip
    ├── test2017.zip
    ├── images/train2017/*.jpg
    ├── images/val2017/*.jpg
    ├── images/test2017/*.jpg         # 仅解压了 test2017 时有
    ├── labels/train2017/*.txt
    ├── labels/val2017/*.txt
    ├── train.csv
    ├── val.csv
    ├── predict.csv
    └── map_class_id_to_class_name.csv
```

## CSV

`train.csv` / `val.csv`：

```text
path_img,path_label_detect_yolo
images/train2017/000000000009.jpg,labels/train2017/000000000009.txt
```

标签是 YOLO 检测格式，每行 `class_id cx cy w h`。

`predict.csv`：

```text
path_img
images/val2017/000000000139.jpg
```

`map_class_id_to_class_name.csv`：

```text
class_id,class_name
0,person
```

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.coco.COCODataModule
  init_args:
    dataset_dir: datasets/COCO/coco
    train_csv_paths: [datasets/COCO/coco/train.csv]
    val_csv_paths: [datasets/COCO/coco/val.csv]
    test_csv_paths: [datasets/COCO/coco/val.csv]
    predict_csv_paths: [datasets/COCO/coco/predict.csv]
    map_class_id_to_class_name: datasets/COCO/coco/map_class_id_to_class_name.csv
```
