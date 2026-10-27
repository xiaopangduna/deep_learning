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
