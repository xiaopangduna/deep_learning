# MobileNetV2 分类转换

容器内在本目录执行。参数与 OpenDLA 调度脚本一致，省略 `-a` 和 `-d`：本目录就是模型目录，分类后处理不进 IPU。

灰度和 RGB 用不同的 `-c` 指定。

```bash
cd exports/sigmastar/mobilenet_v2

# 灰度 eye_v008_gray，输入 1x1x224x224
bash convert.sh \
  -c config/classification_gray.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# RGB eye_v007，输入 1x3x224x224
bash convert.sh \
  -c config/classification.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# 仿真已有的浮点模型。须先转换，且对得上当天的 release 目录
bash convert.sh \
  -c config/classification_gray.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s true
```

`-p` 不写时用上面这条相对路径。`-s` 不写时为 `false`。产物在 `release/<芯片>_<日期>/`。

灰度预处理与训练验证一致：LetterBox 224、单通道、像素除以 255。`input_config_gray.ini` 里对应 `mean=0`、`std_value=255`。换模型时改对应 cfg 的 `Model_LIST` 和 `SAVE_NAME_LIST`，并把 ONNX 放到 `onnx/`。

## 与板端输出比对

同一张图比一次模型输出。板上跑 `.img`，PC 上跑对应 ONNX，再比两份 NPZ。

`dump_img_output_from_image` 在板子上跑。源码在 `deploy_percept/apps/dump_img_output_from_image`，构建见该目录 README。`--model` 用本目录 `release/<芯片>_<日期>/` 里的 `.img`，名字是 cfg 的 `SAVE_NAME_LIST`，例如灰度 `v012_gray.img`、RGB `mobilenet_v2_iford.img`。`--image` 必须和后面 ONNX 用的是同一个文件。图已经是模型 `H×W` 时原样送入，否则 `INTER_LINEAR`。输出写成 float32（元素 × `fScalar`），不做 softmax，也不转置。`--out` 默认 `dump_board.npz`。

```bash
cd /customer/deploy_percept
./bin/dump_img_output_from_image \
  --model ./v012_gray.img \
  --image ./input.png \
  --out ./dump_board.npz
```

把 `dump_board.npz` 拷回本目录。`dump_onnx_output_from_image.py` 用 ONNX 跑同一张图。不写 `--out` 时，输出文件是 ONNX 的主文件名，例如 `onnx/v012.onnx` 写成 `v012.npz`。`compare_img_and_onnx_dumps.py` 读这两份 NPZ：`--board` 是板端那份，`--onnx` 是 ONNX 那份。

`--mean` 和 `--std` 成对使用，公式是 `(像素 - mean) / std`。两个都不写时用 RGB 的 `123.68,116.28,103.53` 和 `58.395,57.12,57.375`。灰度是 1 个通道，这组 3 个数对不上，要写成 `--mean 0 --std 255`。像素原值写 `--mean none --std none`。只写其中一个，或一边是数字、另一边是 `none`，脚本退出。

```bash
cd exports/sigmastar/mobilenet_v2

# 灰度：(x - 0) / 255
python dump_onnx_output_from_image.py \
  --model onnx/v012.onnx \
  --image path/to/input.png \
  --mean 0 \
  --std 255

# RGB：可省略 --mean 和 --std
python dump_onnx_output_from_image.py \
  --model onnx/mobilenet_v2.onnx \
  --image dog.jpg

python compare_img_and_onnx_dumps.py \
  --board dump_board.npz \
  --onnx mobilenet_v2.npz
```

`compare_img_and_onnx_dumps.py` 先对文件哈希、是否 resize、模型宽高。输出 shape 不一致就停并打印两边 shape，不做转置。shape 一致时打印误差。退出码 0 只表示两份文件对得上，数值是否接近看打印结果。

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
