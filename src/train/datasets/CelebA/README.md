# CelebA

DataModule 还没接。下载有两条途径，先走第一条。图像版权不在发布方，用途限于非商业研究和教学。论文是 ICCV 2015 的 *Deep Learning Face Attributes in the Wild*。

## 首要途径：下载整个文件夹

打开 [CelebA 网盘目录](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)，下载整个文件夹，不要逐个点文件。入口也列在 [CelebA 页面](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。带 `resourcekey` 才能打开。

浏览器整包下载得到若干 zip，文件名不固定。可能是 `CelebA-<时间>Z-<卷数>-<序号>.zip`，也可能是 `drive-download-<时间>Z-<卷数>-<序号>.zip`。时间戳、分卷方式和每一卷的字节数每次都会变。这次是 15 卷，`drive-download-20261125T160343Z-1-001.zip` 到 `-015.zip`。每一卷都能单独解开。成员直接是 `README.txt`、`Anno/`、`Eval/`、`Img/`，没有再包一层目录名。这一层解开后还是压缩包和标注文本，没有散落的 jpg。图片在 `Img/img_align_celeba.zip` 里，要再解一次。

放到 `datasets/CelebA/` 后，解进 `datasets/CelebA/celeba/`。外层 zip 用目录里的 `*.zip`，不要写死某一次的文件名：

```bash
mkdir -p datasets/CelebA/celeba
for z in datasets/CelebA/*.zip; do
  unzip -n -q "$z" -d datasets/CelebA/celeba
done
```

数据目录到 `datasets/CelebA/celeba/` 就够了，里面直接是 `Anno/`、`Eval/`、`Img/` 和 `README.txt`。若某次下载的成员顶层多了一层文件夹名（例如 `CelebA/`），再把那一层拿掉；这次没有这一层，下面不用执行：

```bash
if [ -d datasets/CelebA/celeba/CelebA ]; then
  mv datasets/CelebA/celeba/CelebA/* datasets/CelebA/celeba/
  rmdir datasets/CelebA/celeba/CelebA
fi
```

然后解对齐 JPEG。`img_align_celeba/` 会落到 `Img/` 下，里面是 `000001.jpg` 到 `202599.jpg`，共 202599 张，宽 178、高 218：

```bash
unzip -n -q datasets/CelebA/celeba/Img/img_align_celeba.zip \
  -d datasets/CelebA/celeba/Img
```

`Img/img_celeba.7z/` 和 `Img/img_align_celeba_png.7z/` 在网盘里就是文件夹，所以解开后也是文件夹，不是单个 `.7z` 文件。两套都是这 202599 张脸，只是格式不同，用到再解。

- `img_celeba.7z/`：原图，带背景，没有裁成 178×218。14 卷，`img_celeba.7z.001` 到 `.014`。
- `img_align_celeba_png.7z/`：和对齐 JPEG 同一批裁剪图，改成无损 PNG。16 卷，`img_align_celeba_png.7z.001` 到 `.016`。

属性训练用已经解开的 `img_align_celeba/` 即可。要原图或 PNG 时，在对应文件夹里对 `.001` 执行 `7z x`，后面的卷会按序号一起读入：

```bash
7z x datasets/CelebA/celeba/Img/img_celeba.7z/img_celeba.7z.001 \
  -odatasets/CelebA/celeba/Img/img_celeba

7z x datasets/CelebA/celeba/Img/img_align_celeba_png.7z/img_align_celeba_png.7z.001 \
  -odatasets/CelebA/celeba/Img/img_align_celeba_png
```

第一层解开大约再占 22GB，对齐 JPEG 再占约 1.4GB。外面的 zip 留着。外层 zip 的文件名和字节数随每次下载变化，不拿来核对。核对下面这些成员。

合在一起的成员：

| 路径 | 字节 | 内容 |
|---|---|---|
| `README.txt` | 6640 | 官方说明 |
| `Anno/list_attr_celeba.txt` | 26721026 | 40 个属性，取值 `1` / `-1` |
| `Anno/identity_CelebA.txt` | 3424458 | 10177 个身份 |
| `Anno/list_bbox_celeba.txt` | 6082035 | 原图上的框 |
| `Anno/list_landmarks_celeba.txt` | 12352635 | 原图上的 5 个关键点 |
| `Anno/list_landmarks_align_celeba.txt` | 12156055 | 对齐图上的 5 个关键点 |
| `Eval/list_eval_partition.txt` | 2836386 | 训练 / 验证 / 测试 |
| `Img/img_align_celeba.zip` | 1443490838 | 202599 张对齐 JPEG |
| `Img/img_celeba.7z/` | 10191072466 | 原图，14 个 7z 分卷 |
| `Img/img_align_celeba_png.7z/` | 11586962073 | 对齐 PNG，16 个 7z 分卷 |

7z 分卷里，前几卷各 734003200 字节。原图末卷 `img_celeba.7z.014` 是 649030866 字节，对齐 PNG 末卷 `img_align_celeba_png.7z.016` 是 576914073 字节。官方说明写的 218×178 是高×宽，文件里的 JPEG 是宽 178、高 218。

## 次要途径：只下对齐图和标注

整包下不下来时，用脚本只下对齐 JPEG 包和 6 个标注文本。原图和 PNG 的 7z 没有单文件直链，仍走上一节的文件夹。

```bash
python datasets/CelebA/download_celeba.py
python datasets/CelebA/download_celeba.py --output /data/celeba
```

字节数和 MD5 都与下表一致时跳过。半截文件按区间续传，下完再核对。对齐图像包不带 Range 的整段请求会返回配额网页，脚本改用区间拉取。

| 相对 `celeba/` 的路径 | 字节 | MD5 | 文件号 |
|---|---|---|---|
| `Img/img_align_celeba.zip` | 1443490838 | `00d2c5bc6d35e252742224ab0c1e8fcb` | `0B7EVK8r0v71pZjFTYXZWM3FlRnM` |
| `Anno/list_attr_celeba.txt` | 26721026 | `75e246fa4810816ffd6ee81facbd244c` | `0B7EVK8r0v71pblRyaVFSWGxPY0U` |
| `Anno/identity_CelebA.txt` | 3424458 | `32bd1bd63d3c78cd57e08160ec5ed1e2` | `1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS` |
| `Anno/list_bbox_celeba.txt` | 6082035 | `00566efa6fedff7a56946cd1c10f1c16` | `0B7EVK8r0v71pbThiMVRxWXZ4dU0` |
| `Anno/list_landmarks_align_celeba.txt` | 12156055 | `cc24ecafdb5b50baae59b03474781f8c` | `0B7EVK8r0v71pd0FJY3Blby1HUTQ` |
| `Anno/list_landmarks_celeba.txt` | 12352635 | `063ee6ddb681f96bc9ca28c6febb9d1a` | `0B7EVK8r0v71pTzJIdlJWdHczRlU` |
| `Eval/list_eval_partition.txt` | 2836386 | `d32c9cbf5e040fd4025c592c306e6668` | `0B7EVK8r0v71pY0NSMzRuSXJEVkk` |

直链形式与 LFWA 的图像包相同，把文件号换进去即可：

```text
https://drive.usercontent.google.com/download?id=<文件号>&export=download&confirm=t
```

## 目录

```text
datasets/CelebA/
├── README.md
├── download_celeba.py
├── celeba_attr_to_csv.py
├── drive-download-<时间>Z-1-001.zip
├── ...
├── drive-download-<时间>Z-1-015.zip
└── celeba/
    ├── README.txt
    ├── Anno/
    │   ├── list_attr_celeba.txt
    │   ├── identity_CelebA.txt
    │   ├── list_bbox_celeba.txt
    │   ├── list_landmarks_celeba.txt
    │   └── list_landmarks_align_celeba.txt
    ├── Eval/
    │   └── list_eval_partition.txt
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── Img/
        ├── img_align_celeba.zip
        ├── img_align_celeba/000001.jpg
        ├── img_celeba.7z/img_celeba.7z.001
        └── img_align_celeba_png.7z/img_align_celeba_png.7z.001
```

`Anno/`、`Eval/`、`Img/` 和三个 CSV 都在 `celeba/`。实验 YAML 写的就是这一层。某次下载若多包了一层目录名，先按上面的 `if` 挪平，再写 CSV。

`datasets/CelebA/` 留在项目里，放这份 README 和脚本。图像、压缩包和标注不进 Git。

## 软链接

数据在别的磁盘时，把整个数据目录链成 `celeba`。`datasets/CelebA` 本身不要换成软链接。

```bash
ln -s /data/celeba datasets/CelebA/celeba
```

`/data/celeba` 里面就是 `Anno/`、`Eval/`、`Img/` 和 `README.txt`。浏览器下的 15 个 zip 仍放在 `datasets/CelebA/`。

## 标签

训练用 `list_attr_celeba.txt` 的 40 个属性。第一行是图像数 `202599`，第二行是属性名，之后每行是 `<图像名> <40 个取值>`。取值 `1` 表示有，`-1` 表示没有。名字与 LFWA 的 40 个属性相同。下表的中文只供对照。

| 序号 | 名称 | 中文 |
|---|---|---|
| 1 | `5_o_Clock_Shadow` | 胡茬 |
| 2 | `Arched_Eyebrows` | 拱形眉 |
| 3 | `Attractive` | 有吸引力 |
| 4 | `Bags_Under_Eyes` | 眼袋 |
| 5 | `Bald` | 秃顶 |
| 6 | `Bangs` | 刘海 |
| 7 | `Big_Lips` | 厚嘴唇 |
| 8 | `Big_Nose` | 大鼻子 |
| 9 | `Black_Hair` | 黑发 |
| 10 | `Blond_Hair` | 金发 |
| 11 | `Blurry` | 模糊 |
| 12 | `Brown_Hair` | 棕发 |
| 13 | `Bushy_Eyebrows` | 浓眉 |
| 14 | `Chubby` | 圆胖 |
| 15 | `Double_Chin` | 双下巴 |
| 16 | `Eyeglasses` | 戴眼镜 |
| 17 | `Goatee` | 山羊胡 |
| 18 | `Gray_Hair` | 灰发 |
| 19 | `Heavy_Makeup` | 浓妆 |
| 20 | `High_Cheekbones` | 高颧骨 |
| 21 | `Male` | 男性 |
| 22 | `Mouth_Slightly_Open` | 嘴微张 |
| 23 | `Mustache` | 八字胡 |
| 24 | `Narrow_Eyes` | 细长眼 |
| 25 | `No_Beard` | 无胡须 |
| 26 | `Oval_Face` | 鹅蛋脸 |
| 27 | `Pale_Skin` | 肤色苍白 |
| 28 | `Pointy_Nose` | 尖鼻 |
| 29 | `Receding_Hairline` | 发际线后移 |
| 30 | `Rosy_Cheeks` | 脸颊红润 |
| 31 | `Sideburns` | 鬓角 |
| 32 | `Smiling` | 微笑 |
| 33 | `Straight_Hair` | 直发 |
| 34 | `Wavy_Hair` | 波浪发 |
| 35 | `Wearing_Earrings` | 戴耳环 |
| 36 | `Wearing_Hat` | 戴帽子 |
| 37 | `Wearing_Lipstick` | 涂口红 |
| 38 | `Wearing_Necklace` | 戴项链 |
| 39 | `Wearing_Necktie` | 戴领带 |
| 40 | `Young` | 年轻 |

`list_eval_partition.txt` 没有表头，每行 `<图像名> <划分>`。`0` 训练 162770 张，`1` 验证 19867 张，`2` 测试 19962 张。三个划分之间身份不重叠。

`identity_CelebA.txt` 没有表头，每行 `<图像名> <身份编号>`，共 10177 个身份。

`list_bbox_celeba.txt` 的框在原图上，顺序是 `x_1 y_1 width height`，左上角加宽高。不要拿这组坐标去标对齐后的 178×218 图。

两个关键点文件都是 5 点：左眼、右眼、鼻、左嘴角、右嘴角，各一个 `x y`。`list_landmarks_celeba.txt` 用原图坐标，`list_landmarks_align_celeba.txt` 用对齐图坐标。

这些文本写成和 LFWA 相同的框表。属性列是 `attr_` 加属性名，`1` 保持为 `1`，`-1` 写成 `0`。划分用 `list_eval_partition.txt`：`0` 进 `train.csv`，`1` 进 `val.csv`，`2` 进 `test.csv`。图已经是对齐的脸，框是整张图。`path_img` 相对 `datasets/CelebA/celeba/`。三个 CSV 也写在这一层。脚本若发现标注在多出来的 `CelebA/` 里，会改用那一层；实验 YAML 仍指向 `datasets/CelebA/celeba/`。

```bash
python datasets/CelebA/celeba_attr_to_csv.py
python datasets/CelebA/celeba_attr_to_csv.py --overwrite
```

三个 CSV 都已存在时跳过写入，但仍检查属性表、划分和身份。`--overwrite` 在标注换过之后重写。实验配置是 `configs/experiments/face_attr_CelebA_40_mobilenet_v2.yaml`，用 `AttributeFromDetectBoxDataModule`，分类头 40 维。

```bash
python scripts/train.py fit --config configs/experiments/face_attr_CelebA_40_mobilenet_v2.yaml
```

```text
path_img,cx,cy,w,h,img_w,img_h,attr_5_o_Clock_Shadow,...,attr_Young
Img/img_align_celeba/000001.jpg,0.5,0.5,1,1,178,218,0,...,1
```

要少训几个属性，在实验 YAML 的 `attribute_columns` 里点名，不用改 CSV。
