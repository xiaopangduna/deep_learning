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

只覆盖本目录的 MobileNetV2。板端用 `deploy_percept` 的 `img_tensor_dump_sigmastar` 对同一张已经缩好的图写出 `dump_board.npz`，拷回这里再比。

```bash
cd exports/sigmastar/mobilenet_v2

# 灰度：与 input_config_gray.ini 的 std_value=255 对齐
python eval_onnx_dump.py \
  --model onnx/v012.onnx \
  --image path/to/input.png \
  --out dump_onnx.npz \
  --value-scale 0.00392156862745098

# RGB ImageNet：mean/std 在 input_config.ini。ONNX 若已含归一化，保持 --value-scale 1
python eval_onnx_dump.py \
  --model onnx/mobilenet_v2.onnx \
  --image imagenet_10/1.jpg \
  --out dump_onnx.npz \
  --value-scale 1

python compare_dumps.py \
  --board dump_board.npz \
  --onnx dump_onnx.npz
```

`compare_dumps.py` 先对文件哈希、是否 resize、模型宽高。输出 shape 不一致就停并打印两边 shape，不做转置。
