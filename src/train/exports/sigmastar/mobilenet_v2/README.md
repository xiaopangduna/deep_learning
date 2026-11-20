# SigmaStar 部署 MobileNetV2

## MobileNetV2 onnx 转 img

### 模型转换

容器内在本目录执行。参数与 OpenDLA 调度脚本一致，省略 `-a` 和 `-d`：本目录就是模型目录，分类后处理不进 IPU。

灰度和 RGB 用不同的 `-c` 指定。

```bash
cd exports/sigmastar/mobilenet_v2

# 单通道，1x1x224x224，读 config/classification_gray.cfg
bash convert.sh \
  -c config/classification_gray.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# 三通道，1x3x224x224，读 config/classification.cfg
bash convert.sh \
  -c config/classification.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false
```

`-p` 不写时用上面这条相对路径。`-s` 不写时为 `false`；写成 `true` 时只仿真当天 `release/` 里已有的 `*_float.sim`，不再重新转换。产物在 `release/<芯片>_<日期>/`。`convert.sh` 读的 ONNX 是 `onnx/${Model_LIST}.onnx`。

### 参数配置

换模型时改对应 cfg 的 `Model_LIST`、`SAVE_NAME_LIST`，并把 ONNX 放到 `onnx/`。`INPUT_INI_LIST` 指向的 ini 里，`inputs` / `outputs` 要和这份 ONNX 的张量名一致。导出配置和这里的 ini 都用 `input`、`output`。

几何和标准化分开：

- resize、letterbox、ImageNet 中心裁切在跑图程序里做（板端 `--prep`，ONNX 脚本用同一个 `--prep`），不写进 `.img`。裁框在这之前：`--box-scale > 0` 且 CSV 有 `cx,cy,w,h` 时，先按归一化框外扩再做 `--prep`。
- `(像素 - mean) / std` 写进 `.img`。灰度 `input_config_gray.ini` 是 `mean=0`、`std_value=255`，对应训练验证的 LetterBox 224、单通道。RGB `input_config.ini` 是 `123.68,116.28,103.53` 和 `58.395:57.12:57.375`。ONNX 侧用同一组数。

灰度裁框模型的校准图放在 `quant_data/`。每张是裁框后再 LetterBox 到 224、填充 114 的 uint8，不做 `/255`。`mobilenet_gray_pre.py` 在校准里再做一次 LetterBox、灰度和 `/255`。图已经是 224×224 时第二次 LetterBox 的比例是 1，内容不变。不要把还没 letterbox 的原框，或直接拉成正方形的图放进这个目录。

## 单张图精度验证：ONNX 与 img

同一张图比一次模型输出。板上跑 `.img`，PC 上跑对应 ONNX，再比两份 NPZ。`--image` 必须是同一个文件。图已经是模型 `H×W` 时原样送入，否则 `INTER_LINEAR`。输出写成 float32（元素 × `fScalar`），不做 softmax，也不转置。

### img 运行结果

`dump_img_output_from_image` 在板子上跑。源码在 `deploy_percept/apps/dump_img_output_from_image`，构建见该目录 README。`--model` 用本目录 `release/<芯片>_<日期>/` 里的 `.img`，名字是 cfg 的 `SAVE_NAME_LIST`。`--out` 默认 `dump_board.npz`。

```bash
cd /customer/deploy_percept
./bin/dump_img_output_from_image \
  --model ./mobilenet_v2_iford.img \
  --image ./dog.jpg \
  --out ./dump_board.npz
```

把 `dump_board.npz` 拷回本目录。

### onnx 运行结果

`verify/single_image_similarity_dump_onnx_npz.py` 用 ONNX 跑同一张图。不写 `--out` 时，输出文件是 ONNX 的主文件名，例如 `onnx/mobilenet_v2.onnx` 写成 `mobilenet_v2.npz`。

`--mean` 和 `--std` 成对使用，公式是 `(像素 - mean) / std`。两个都不写时用 RGB 的 `123.68,116.28,103.53` 和 `58.395,57.12,57.375`。灰度是 1 个通道，这组 3 个数对不上，要写成 `--mean 0 --std 255`。像素原值写 `--mean none --std none`。只写其中一个，或一边是数字、另一边是 `none`，脚本退出。

```bash
cd exports/sigmastar/mobilenet_v2

# RGB：可省略 --mean 和 --std
python verify/single_image_similarity_dump_onnx_npz.py \
  --model onnx/mobilenet_v2.onnx \
  --image dog.jpg

# 灰度：(x - 0) / 255
python verify/single_image_similarity_dump_onnx_npz.py \
  --model onnx/v013.onnx \
  --image path/to/input.png \
  --mean 0 \
  --std 255
```

### 对比 onnx 和 img 结果

`verify/single_image_similarity_compare_board_onnx_npz.py` 读这两份 NPZ：`--board` 是板端那份，`--onnx` 是 ONNX 那份。它先对文件哈希、是否 resize、模型宽高。输出 shape 不一致就停并打印两边 shape，不做转置。shape 一致时打印误差。退出码 0 只表示两份文件对得上，数值是否接近看打印结果。

```bash
python verify/single_image_similarity_compare_board_onnx_npz.py \
  --board dump_board.npz \
  --onnx mobilenet_v2.npz
```

分类输出是 `(1, 类别数)` 的 logit。余弦在 -1 到 1 之间，大于 **0.99** 就可以认为这张图的转换正常。余弦只看方向，所以同时看 `max_abs`：它若到了几十，多半是整体缩放错了，余弦仍可能接近 1。`dog.jpg` 对 `mobilenet_v2.onnx` 的一次结果：

```text
output_0 shape=(1, 1000) max_abs=0.98135 mean_abs=0.21693 rmse=0.279152 cosine=0.995501 p99=0.752575 rel=0.100847
```

| 指标 | 含义 |
|---|---|
| cosine | 两边向量的方向。0.9955 表示对齐；没做归一化时只有 0.067 |
| max_abs | 差得最多的一个类别。0.98，logit 大约在 -9 到 13 |
| p99 | 99% 的类别差距不超过这个值 |
| mean_abs | 每个类别的平均绝对差 |
| rmse | 差的均方根 |
| rel | 平均绝对差除以 ONNX 输出的平均幅度 |

这次 top-5 都是 249, 248, 250, 537, 235，top-1 的分数几乎相同。这是 INT16 量化误差，分类结果不变。换几张图仍然大于 0.99，这条转换链路就是稳的。

## 多图 ACC 精度验证：ONNX 与 img

两边各写一份 JSON。样本里的 `class_id` 是 CSV 原值，`pred_id` 是模型输出下标。板端的 `acc1` / `acc5` 在 `meta`，ONNX 的在 `summary`，都是这两个数直接比较，不做训练标签映射。

整图且 `class_id` 就是模型类别下标时，直接读这两处。裁框实验的原始 id 和训练 id 往往不一致，用 [`scripts/score_classifier_dump.py`](../../../scripts/score_classifier_dump.py) 按 `class_groups` 重算。指标含义、映射写法和哪些样本计入 `skip` 都写在该脚本开头。top-1 相差小于 1 个点，认为这批评测对齐。

几何用 `--prep`，名字和板端相同。板端默认是 `letterbox`。`imagenet`：短边缩到 `floor(224/0.875)=256`，再中心裁 224。`letterbox` 保持宽高比，空处填 114。`resize` 直接拉到 224。像素是 `(像素 - mean) / std`。不写 `--mean` 和 `--std` 时用 `input_config.ini` 的 `123.68,116.28,103.53` 和 `58.395,57.12,57.375`。`.img` 里这步已经做了，ONNX 要在脚本里做。灰度必须写成 `--mean 0 --std 255`。板端那份如果是 NV12 打包，logit 不会和 ONNX 逐元素相同。

推理优先 CUDA。脚本会先加载 uv 环境里 `nvidia-*` 包的 CUDA 12 / cuDNN 9，不看 shell 的 `LD_LIBRARY_PATH`（那边如果还是 CUDA 11.8，对不上 onnxruntime-gpu 1.23）。CUDA 不可用时退回 CPU。启动时打印 `provider:`。

### 整图

`classifier_dump_sigmastar` 在板子上跑。源码在 `deploy_percept/apps/classifier_dump_sigmastar`，构建和参数见该目录 README。ImageNet 用 `--prep imagenet`，不要加 `--box-scale`。

```bash
cd /customer/deploy_percept
./bin/classifier_dump_sigmastar \
  --model ./mobilenet_v2_iford.img \
  --csv ./val.csv \
  --prep imagenet \
  --out ./mobilenet_v2_iford.json
```

JSON 拷回 `tmp/mobilenet_v2_iford.json`。

`verify/multi_image_accuracy_dump_onnx_json.py` 按 CSV 跑 ONNX，写成和板端相同的 JSON。`--limit N` 只跑 CSV 前 N 行。没有成功样本时退出码为 1。不写 `--box-scale`，或值 `<= 0`，就是整图。

```bash
cd exports/sigmastar/mobilenet_v2

python verify/multi_image_accuracy_dump_onnx_json.py \
  --model onnx/mobilenet_v2.onnx \
  --csv datasets/IMAGENET/imagenet/val.csv \
  --prep imagenet \
  --out ../../../tmp/mobilenet_v2.json
```

ImageNet 的 `class_id` 就是类别下标，直接读两份 JSON 的 `acc1` / `acc5`。2026-11-17 的验证集 5 万张，ONNX 与板端 top-1 差 0.46 个点，top-5 差 0.21 个点。

PyTorch 官方 `MobileNet_V2_Weights.IMAGENET1K_V1`（`mobilenet_v2-b0353104.pth`）在[分类权重表](https://docs.pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)的 Acc@1 是 **71.878**，Acc@5 是 90.286。

2026-11-18 用这份权重和 `image_classifiter_IMAGE_NET_mobilenet_v2.yaml` 跑验证集 5 万张。`transform_val` 的 `Resize` 是短边 256、保持宽高比，再中心裁 224。`val_acc` 为 **71.850%**（0.71850），`val_loss` 为 1.148。比官方 Acc@1 低 0.028 个点。把 `Resize` 写成 `[256, 256]` 会拉成正方形，同一次对照只有 71.208%。

```bash
python scripts/train.py validate \
  --config configs/experiments/image_classifiter_IMAGE_NET_mobilenet_v2.yaml
```

| | top-1 | top-5 |
|---|---:|---:|
| PyTorch 官方 | 71.878% | 90.286% |
| 本仓库 validate | 71.850% | 未记录 |
| ONNX | 70.722% | 89.822% |
| 板端 | 70.260% | 89.616% |

### 裁框

CSV 除了 `path_img,class_id`，还要有归一化的 `cx,cy,w,h`。`--box-scale` 与训练 YAML 的 `box_scale` 相同；v002 是 `1.0`。板端和 ONNX 都是先外扩裁切，再 `--prep letterbox`。不写 `--box-scale` 会把整张图送进模型，和训练对不上。样本里仍保存 CSV 的原始 `class_id` 和还没乘 `box_scale` 的框。

```bash
cd /customer/deploy_percept
./bin/classifier_dump_sigmastar \
  --model ./v013_gray.img \
  --csv ./test_v302.csv \
  --box-scale 1.0 \
  --prep letterbox \
  --out ./v013_gray.json
```

```bash
cd exports/sigmastar/mobilenet_v2

python verify/multi_image_accuracy_dump_onnx_json.py \
  --model onnx/v013.onnx \
  --csv /path/to/test_v302.csv \
  --prep letterbox \
  --box-scale 1.0 \
  --mean 0 \
  --std 255 \
  --out ../../../tmp/v013_onnx.json
```

`--csv` 用板端那份表。JSON 里的 `acc1` / `acc5` 仍是原始 `class_id` 对 `pred_id`。v002 的 `class_groups` 是 `0: [0]`、`1: [2]`，原始 `class_id` 2 对应 `pred_id` 1，原始 `1` 不在组里。这个对应不在 dump 里做。

### 评估

在 `train` 仓库根目录跑 `scripts/score_classifier_dump.py`。`--class-groups` 写成和 YAML 一样的组，训练 id 在冒号左边。`--dump` 可重复，同一条映射打在板端和 ONNX 两份 JSON 上。脚本开头写了映射语法、`skip`、`acc`、`acc_macro`、混淆矩阵、recall 和 precision。

```bash
python scripts/score_classifier_dump.py \
  --dump tmp/v013_gray.json \
  --dump tmp/v013_onnx.json \
  --class-groups '0:0;1:2'
```

2026-11-20，`test_v302` 映射后留下 5553 条，`skip` 67 条原始 `class_id` 1。全体准确率两边都是 **0.965964**（5364/5553）。`acc_macro` 板端 **0.961427**，ONNX **0.959327**。类 0 recall 是 4461/4607 对 4466/4607，类 1 recall 是 903/946 对 898/946。正确总数相同。

## 官方 ONNX 与 SigmaStar ONNX

官方 ONNX 用 `configs/experiments/image_classifiter_IMAGE_NET_mobilenet_v2.yaml` 导出，对比文件是 `mobilenet_v2_office.onnx`。

```bash
python scripts/train.py export \
  --config configs/experiments/image_classifiter_IMAGE_NET_mobilenet_v2.yaml \
  --export_format onnx
```

SigmaStar ONNX 是 SDK 自带的分类模型，对比文件是 `mobilenet_v2.onnx`。本目录转换也读这个文件：`config/classification.cfg` 的 `Model_LIST` 是 `mobilenet_v2`。

```text
exports/sigmastar/sdk/IPU_SDK_Release_25121210/OpenDLAModel_20250721/classification/mobilenet_v2/onnx/mobilenet_v2.onnx
```

两边都是 100 个节点，输入 `input` `[1, 3, 224, 224]`，输出 `output` `[1, 1000]`。前 98 个算子顺序相同。最后一步是 Flatten 对 Reshape，参数量 3,487,818 对 3,487,820，多出来的是 Reshape 的形状常量。这点差按简化器版本不同来看，不用管。2026-11-17 用 `scripts/compare_onnx_on_image.py` 喂同一张图：

```bash
python scripts/compare_onnx_on_image.py \
  mobilenet_v2_office.onnx \
  mobilenet_v2.onnx \
  --image tmp/000000562557.jpg
```

```text
out[0] output vs output  [1, 1000]  max_abs=0.000000e+00 mean_abs=0.000000e+00  PASS
```

1000 维 logit 逐元素相同。

JSON 来自 ImageNette `datasets/IMAGENETTE/train.csv` 的 9469 张。`verify/multi_image_accuracy_dump_onnx_json.py` 分别跑两个 ONNX，得到 `mobilenet_v2_office.json` 和 `mobilenet_v2.json`。

| | top-1 | top-5 |
|---|---:|---:|
| 官方 | 9.558%（905/9469） | 10.075%（954/9469） |
| SigmaStar | 9.558%（905/9469） | 10.075%（954/9469） |

数值低是因为 CSV 的 `class_id` 是 ImageNette 类别编号，脚本把它直接当成 ImageNet 1000 类下标去对 `pred_id`，没有做 ImageNette 到 ImageNet 的映射。9469 条样本的 `pred_id` 和 1000 维 logit 也逐项相同，logit 最大绝对差为 0。指标和内容都一致，可以认为这两个 ONNX 完全一致。

```bash
python exports/sigmastar/mobilenet_v2/verify/multi_image_accuracy_dump_onnx_json.py \
  --model mobilenet_v2_office.onnx \
  --csv datasets/IMAGENETTE/train.csv

python exports/sigmastar/mobilenet_v2/verify/multi_image_accuracy_dump_onnx_json.py \
  --model mobilenet_v2.onnx \
  --csv datasets/IMAGENETTE/train.csv
```
