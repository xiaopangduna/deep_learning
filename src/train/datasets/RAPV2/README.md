# RAP v2

84928 张监控场景里的行人裁剪图，每张带属性。训练用 `AttributeFromDetectBoxDataModule`。图像来自 [OpenPAR 的 Dropbox 目录 RAPV2](https://www.dropbox.com/scl/fo/boipdmufnsnsvmfdle5um/ACGa3vnoT5vAJVSfTvfB00U/RAPV2?rlkey=ftwdjxqo5l1dfnshy97ej8kka&e=1&dl=0)。论文是 IEEE TIP 2019 的 *A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenarios*。

## 图像

网盘里的图像包文件名可能是 `RAP_dataset.zip`，浏览器重复下载时会变成 `RAP_dataset(1).zip`。当前这份是 `RAP_dataset(1).zip`，4.0GB，84928 张 PNG，外加一个空的 `RAP_dataset/.gitsave`。密码在整包里的 `RAP_unzip_code.txt`，RAP2.0 那一行是 `casia_crise_195610`。

解到 `datasets/RAPV2/`，`RAP_dataset/` 会落在这一层：

```bash
unzip -P 'casia_crise_195610' \
  "datasets/RAPV2/RAP_dataset(1).zip" \
  -d datasets/RAPV2
```

每张图已经是一个行人。文件名形如 `CAM31-2014-03-18-20140318125002-20140318125546-tarid93-frame2823-line1.png`。宽高不固定。

同目录的 `RAP_annotation.zip` 是 RAP v1 的标注，打开后写着 41585 张图。它和 `tmp/PAR_public_benchmark_datasets (10.3 GB)/RAP/RAP_annotation.zip` 是同一个文件。密码是解压码里 RAP1.0 那一行 `casia_cripac_isee_5610`。v2 训练用下面的 pkl。

## 标注

OpenPAR 把 v2 标注单独放成 [rapv2_expand.pkl](https://www.dropbox.com/scl/fi/q4v2fjb1o5823qmaw6979/rapv2_expand.pkl?rlkey=wygnhxfv2z3ur8wa6or8lz5w8&dl=0)，63MB。下到 `datasets/RAPV2/rapv2_expand.pkl`。

这份 pkl 是 `easydict.EasyDict`。和 CSV 有关的字段：

| 字段 | 形状 | 内容 |
|---|---|---|
| `image_name` | 84928 | 与图像包一一对应的文件名 |
| `attr_name` | 54 | 实验属性的英文短名。性别写的是 `Femal` |
| `label` | 84928 × 54 | `0` / `1`。`Femal` 另有 161 个 `2`，表示不确定 |
| `expand_attributes` | 44 | 上衣、下衣、鞋子的颜色，以及遮挡位置和遮挡原因 |
| `expand_label` | 84928 × 44 | `0` / `1` |
| `partition` | 5 组 | 每组 `train` 50957、`val` 16986、`test` 16985，三份互斥并盖住全部图像 |

`expand_attr_name` 从鞋子颜色开始与 `expand_attributes` 的短语对不上。CSV 用短语，不用这列短名。

5 组划分彼此不同。默认写第 0 组，也就是 OpenPAR 读取的 `partition["train"][0]`。`Femal` 的 `2` 写成 `0`。

```bash
python datasets/RAPV2/rapv2_attr_to_csv.py
python datasets/RAPV2/rapv2_attr_to_csv.py --split-index 1 --overwrite
```

三个 CSV 都已存在时跳过写入，但仍检查 pkl。`--split-index` 换成另一组时要加 `--overwrite`，文件名里不带组号。

`path_img` 相对 `datasets/RAPV2/`。框是整张图。98 个属性在同一张表里：前 54 列是实验属性，后 44 列是颜色和遮挡。

```text
path_img,cx,cy,w,h,img_w,img_h,attr_hs-BaldHead,...,attr_action-Other,attr_Black_upper_clothing,...,attr_Occluded_by_other_factors
RAP_dataset/CAM31-2014-03-18-20140318125002-20140318125546-tarid93-frame2823-line1.png,0.5,0.5,1,1,宽,高,0,...,0,0,...,0
```

## 目录

```text
datasets/RAPV2/
├── README.md
├── rapv2_attr_to_csv.py
├── RAP_dataset(1).zip
├── RAP_annotation.zip
├── rapv2_expand.pkl
├── train.csv
├── val.csv
├── test.csv
└── RAP_dataset/
    └── CAM31-2014-03-18-20140318125002-20140318125546-tarid93-frame2823-line1.png
```

`datasets/RAPV2/` 留在项目里，放这份 README 和脚本。图像、压缩包、pkl 和 CSV 不进 Git。

## 软链接

数据在别的磁盘时，把行人图目录链成 `RAP_dataset`。pkl 和 CSV 留在 `datasets/RAPV2/`。

```bash
ln -s /data/RAP_dataset datasets/RAPV2/RAP_dataset
```

`/data/RAP_dataset` 里面直接是 png。pkl 和 CSV 仍放在 `datasets/RAPV2/`。

## 标签

下表的中文只供对照。列名是 `attr_` 加 `attr_name`，短名里的连字符保留。

| 序号 | 名称 | 中文 |
|---|---|---|
| 1 | `hs-BaldHead` | 秃头 |
| 2 | `hs-LongHair` | 长发 |
| 3 | `hs-BlackHair` | 黑发 |
| 4 | `hs-Hat` | 帽子 |
| 5 | `hs-Glasses` | 眼镜 |
| 6 | `ub-Shirt` | 衬衫 |
| 7 | `ub-Sweater` | 毛衣 |
| 8 | `ub-Vest` | 背心 |
| 9 | `ub-TShirt` | T 恤 |
| 10 | `ub-Cotton` | 棉衣 |
| 11 | `ub-Jacket` | 夹克 |
| 12 | `ub-SuitUp` | 西装上衣 |
| 13 | `ub-Tight` | 紧身上衣 |
| 14 | `ub-ShortSleeve` | 短袖 |
| 15 | `ub-Others` | 其他上衣 |
| 16 | `lb-LongTrousers` | 长裤 |
| 17 | `lb-Skirt` | 半身裙 |
| 18 | `lb-ShortSkirt` | 短裙 |
| 19 | `lb-Dress` | 连衣裙 |
| 20 | `lb-Jeans` | 牛仔裤 |
| 21 | `lb-TightTrousers` | 紧身裤 |
| 22 | `shoes-Leather` | 皮鞋 |
| 23 | `shoes-Sports` | 运动鞋 |
| 24 | `shoes-Boots` | 靴子 |
| 25 | `shoes-Cloth` | 布鞋 |
| 26 | `shoes-Casual` | 休闲鞋 |
| 27 | `shoes-Other` | 其他鞋 |
| 28 | `attachment-Backpack` | 双肩包 |
| 29 | `attachment-ShoulderBag` | 单肩包 |
| 30 | `attachment-HandBag` | 手提包 |
| 31 | `attachment-Box` | 箱子 |
| 32 | `attachment-PlasticBag` | 塑料袋 |
| 33 | `attachment-PaperBag` | 纸袋 |
| 34 | `attachment-HandTrunk` | 手拉箱 |
| 35 | `attachment-Other` | 其他附属物 |
| 36 | `AgeLess16` | 16 岁及以下 |
| 37 | `Age17-30` | 17 到 30 岁 |
| 38 | `Age31-45` | 31 到 45 岁 |
| 39 | `Age46-60` | 46 到 60 岁 |
| 40 | `Femal` | 女性 |
| 41 | `BodyFat` | 胖 |
| 42 | `BodyNormal` | 普通体型 |
| 43 | `BodyThin` | 瘦 |
| 44 | `Customer` | 顾客 |
| 45 | `Employee` | 店员 |
| 46 | `action-Calling` | 打电话 |
| 47 | `action-Talking` | 交谈 |
| 48 | `action-Gathering` | 聚集 |
| 49 | `action-Holding` | 持物 |
| 50 | `action-Pushing` | 推 |
| 51 | `action-Pulling` | 拉 |
| 52 | `action-CarryingByArm` | 抱在臂上 |
| 53 | `action-CarryingByHand` | 提在手里 |
| 54 | `action-Other` | 其他动作 |

后 44 列的列名是 `attr_` 加英文短语，空白换成下划线。顺序与 `expand_attributes` 一致：12 种上衣颜色、12 种下衣颜色、12 种鞋子颜色、4 个遮挡位置、4 个遮挡原因。颜色依次是黑、白、灰、红、绿、蓝、银、黄、棕、紫、粉、橙。遮挡位置是左、右、上、下。遮挡原因是环境、附属物、另一个人、其他。

实验配置是 `configs/experiments/pedestrian_attr_RAPV2_54_mobilenet_v2.yaml`，`attribute_columns` 点名要训的列，当前是前 54 个实验属性。`num_attributes` 和分类头的 `out_features` 与这份名单一样长。换一组属性时复制该文件，改名单并把这两个数改成同样的长度。图已经是行人，框又盖住整张图，`box_scale` 放大于图像外的部分会被裁掉。
